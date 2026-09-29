import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (MessagePassing, GATConv, SAGEConv,
                                 APPNP as APPNPProp)
from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.utils import (add_self_loops, degree, to_networkx,
                                   to_scipy_sparse_matrix, remove_self_loops)
import networkx as nx
import numpy as np
import argparse, json, math, os
import multiprocessing as mp
from collections import defaultdict, OrderedDict
from scipy.stats import wilcoxon, ttest_rel

from torch_geometric.datasets import (Planetoid, WebKB, Actor,
                                       WikipediaNetwork,
                                       HeterophilousGraphDataset)
from torch_geometric.utils import (add_self_loops, degree, to_networkx,
                                    to_scipy_sparse_matrix, remove_self_loops,
                                    to_undirected, get_laplacian)
import time
import urllib.request
from torch_geometric.data import Data as _PygData
from torch_geometric.utils import softmax as pyg_softmax


# ═══════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
#
# Bloco 1 — Varredura de normalizações (CGNN × nctm × norm):
#   CGNN : baseline curvado, sem separação de raiz
#
# Bloco 2 — Cadeia de ablação ACN-GCN (nctm=sigmatemp, sweep norm):
#   ACN-GCN-noroot : curvatura ponderada, SEM source separation
#   ACN-GCN        : curvatura ponderada, COM source separation (W_source)
#
# Baselines: GCN, GAT, GraphSAGE, H2GCN, GPRGNN, APPNP, MixHop, CurvGN, FAGCN, ACM-GCN, A2GCN
# ═══════════════════════════════════════════════════════════════

DATASETS_CONFIG = OrderedDict([
    # Vou rodar apenas com esses aqui    
    ('Texas',     {'type': 'WebKB'}),
    ('Wisconsin', {'type': 'WebKB'}),
    ('Cornell',   {'type': 'WebKB'}),
    #('chameleon',      {'type': 'Wikipedia'}), # ~2.2k nós, h≈0.23, 5 classes
    #('squirrel',       {'type': 'Wikipedia'}), # ~5.2k nós, h≈0.22, 5 classes
    #('crocodile',      {'type': 'Wikipedia'}), # ~11.6k nós, h≈0.25, 5 classes
    #('roman-empire',   {'type': 'Hetero'}),    # ~22.6k nós, h≈0.05, 18 classes (Platonov et al., 2024)
    # Chameleon/Squirrel filtrados (Platonov et al., 2023) — sem nós duplicados / leakage.
    # chameleon-filtered: 890 nós, 8.854 arestas, 2.325 feats, 5 classes, 10 splits.
    ('chameleon-filtered', {'type': 'FilteredNPZ', 'npz': 'chameleon_filtered'}),
    ('squirrel-filtered',  {'type': 'FilteredNPZ', 'npz': 'squirrel_filtered'}),
])

SEEDS          = [42, 123, 456, 789, 1024]
# NEGATIVE CURVATURE TRANSFORMATION MODULE
NCTM_ALL       = ['linear', 'sigmoid', 'sigmatemp']
# NORMALIZATION MODULES
NORM_ALL       = ['src', 'dst', 'sym', 'deep_sym', 'alpha', 'acn']

NORM_SWEEP_KEYS = ['CGNN']
ABLATION_KEYS   = ['ACN-GCN-noroot', 'ACN-GCN']
ACN_GCN_KEYS    = NORM_SWEEP_KEYS + ABLATION_KEYS   # compatibilidade
BASELINE_KEYS   = ['GCN', 'GAT', 'GraphSAGE', 'H2GCN',
                   'GPRGNN', 'APPNP', 'MixHop', 'CurvGN',
                   'FAGCN', 'ACM-GCN', 'A2GCN']
ALL_KEYS        = BASELINE_KEYS + ACN_GCN_KEYS

_HET_BASE = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data"

class _NPZGraphDataset:
    """Wrapper mínimo no formato dataset-PyG para os .npz de Platonov et al. (2023).

    As arestas são armazenadas UMA vez (não-direcionadas); load_all_splits chama
    to_undirected, que duplica corretamente para [2, 2E]."""
    def __init__(self, npz_name, root='./data'):
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, f'{npz_name}.npz')
        if not os.path.exists(path):
            url = f'{_HET_BASE}/{npz_name}.npz'
            print(f'  baixando {npz_name}.npz …')
            urllib.request.urlretrieve(url, path)
        raw = np.load(path)
        x  = torch.from_numpy(raw['node_features']).float()          # [N, F]
        y  = torch.from_numpy(raw['node_labels']).long()             # [N]
        ei = torch.from_numpy(raw['edges']).long().t().contiguous()  # [2, E]
        # masks: [num_splits, N] → [N, num_splits], como espera load_all_splits
        tr = torch.from_numpy(raw['train_masks']).bool().t().contiguous()
        va = torch.from_numpy(raw['val_masks']).bool().t().contiguous()
        te = torch.from_numpy(raw['test_masks']).bool().t().contiguous()
        self._data = _PygData(x=x, y=y, edge_index=ei,
                              train_mask=tr, val_mask=va, test_mask=te,
                              num_nodes=x.size(0))
        self.num_features = int(x.size(1))
        self.num_classes  = int(y.max().item()) + 1

    def __len__(self):  return 1
    def __getitem__(self, idx):
        assert idx == 0
        return self._data


def _build_dataset(ds_name):
    t = DATASETS_CONFIG[ds_name]['type']
    if   t == 'Planetoid':   return Planetoid(root=f'./data/{ds_name}', name=ds_name)
    elif t == 'WebKB':       return WebKB(root=f'./data/{ds_name}', name=ds_name)
    elif t == 'Actor':       return Actor(root=f'./data/{ds_name}')
    elif t == 'Wikipedia':   return WikipediaNetwork(root=f'./data/{ds_name}',
                                                     name=ds_name.lower(),
                                                     geom_gcn_preprocess=True)
    elif t == 'Hetero':      return HeterophilousGraphDataset(root=f'./data/{ds_name}',
                                                              name=ds_name)
    elif t == 'FilteredNPZ': return _NPZGraphDataset(DATASETS_CONFIG[ds_name]['npz'])
    raise ValueError(t)

def load_all_splits(ds_name, device):
    """Carrega o dataset uma vez e devolve TODAS as máscaras de split.
    Topologia é constante entre splits → kappa/r_norm computados uma só vez."""
    ds = _build_dataset(ds_name)
    data = ds[0]
    # Datasets Platonov armazenam cada aresta uma vez; garante bidirecional.
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    if data.train_mask.dim() > 1:                       # [N, n_splits]
        n = data.train_mask.shape[1]
        masks = [(data.train_mask[:, j], data.val_mask[:, j], data.test_mask[:, j])
                 for j in range(n)]
    else:
        masks = [(data.train_mask, data.val_mask, data.test_mask)]
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    data = data.to(device)
    masks = [tuple(t.to(device) for t in trio) for trio in masks]
    return ds, data, masks

def compute_curvature_cached(method, data, device, ds_name):
    """Curvatura é topológica → computa uma vez por dataset, cacheia em disco.
    Também serve para reportar o custo de pré-processamento (Revisor 1)."""
    os.makedirs('./cache', exist_ok=True)
    path = f'./cache/kappa_{ds_name}_{method}.pt'
    E = data.edge_index.shape[1]
    if os.path.exists(path):
        k = torch.load(path, map_location='cpu')
        if k.numel() == E:
            print(f'  curvatura {method} ({ds_name}) lida do cache')
            return k.to(device)
        print('  ⚠ cache incompatível (E mudou) — recomputando')
    t0 = time.time()
    kappa = compute_curvature_weights(method, data, device)
    print(f'  curvatura {method} ({ds_name}): {time.time()-t0:.1f}s  (E={E})')
    torch.save(kappa.cpu(), path)
    return kappa.to(device)

def _sig_star(p):
    return '*' if (p == p and p < 0.05) else ' '       # p==p descarta NaN

def select_norm_by_val(ablation_val, model_key='ACN-GCN'):
    """Escolhe a normalização com maior acurácia média de VALIDAÇÃO (sem test-peeking)."""
    best_norm, best_mean = 'acn', -1.0
    for norm, models in ablation_val.items():
        vals = models.get(model_key, [])
        if vals:
            m = float(np.mean(vals))
            if m > best_mean:
                best_mean, best_norm = m, norm
    return best_norm

def _print_significance(baseline_accs, ablation_accs, sel_norm='acn'):
    """Wilcoxon/t pareado: ACN-GCN (norma selecionada por val) vs cada baseline."""
    head = ablation_accs.get(sel_norm, {}).get('ACN-GCN', [])
    if not head:
        return
    _hdr = f'Significância (ACN-GCN + {sel_norm} vs baselines)'
    print(f'\n  {_hdr:─<70}')
    for key in BASELINE_KEYS:
        b = baseline_accs.get(key, [])
        if not b or len(b) != len(head):
            print(f'  {key:<12}: (n incompatível — pulado)'); continue
        s = paired_significance(head, b)
        print(f'  {key:<12}: Δacc={s["mean_diff"]*100:+5.1f}  '
              f'p_wilcoxon={s["p_wilcoxon"]:.4f}{_sig_star(s["p_wilcoxon"])}  '
              f'p_t={s["p_ttest"]:.4f}')

def paired_significance(accs_a, accs_b):
    """Teste pareado entre dois modelos sobre os MESMOS (split, seed).
    Pré-condição: accs_a[k] e accs_b[k] vêm da mesma partição/seed."""
    a, b = np.asarray(accs_a), np.asarray(accs_b)
    diff = a - b
    if np.allclose(diff, 0):
        return dict(mean_diff=0.0, p_ttest=1.0, p_wilcoxon=1.0)
    out = dict(mean_diff=float(diff.mean()),
               p_ttest=float(ttest_rel(a, b).pvalue))
    try:    out['p_wilcoxon'] = float(wilcoxon(a, b).pvalue)
    except ValueError: out['p_wilcoxon'] = float('nan')
    return out

# ═══════════════════════════════════════════════════════════════
# 1. CURVATURA
# ═══════════════════════════════════════════════════════════════

def sinkhorn_log(log_mu, log_nu, C, eps=0.1, n_iter=15):
    log_K = -C / eps
    log_b = torch.zeros_like(log_nu)
    for _ in range(n_iter):
        log_a = log_mu - torch.logsumexp(log_K + log_b.unsqueeze(1), dim=2)
        log_b = log_nu - torch.logsumexp(log_K + log_a.unsqueeze(2), dim=1)
    log_P = log_a.unsqueeze(2) + log_K + log_b.unsqueeze(1)
    return (C * torch.exp(log_P)).sum(dim=[1, 2])

def build_orc_inputs(edge_index, num_nodes):
    import torch_geometric.data as geom_data
    G = to_networkx(geom_data.Data(edge_index=edge_index, num_nodes=num_nodes),
                    to_undirected=True)
    dist_dict = dict(nx.shortest_path_length(G))
    D = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            D[i, j] = dist_dict[i][j] if (i in dist_dict and j in dist_dict[i]) else 4.0
    E = edge_index.shape[1]
    type_u = torch.zeros((E, num_nodes)); type_v = torch.zeros((E, num_nodes))
    C_geo  = torch.zeros((E, num_nodes, num_nodes))
    src, dst = edge_index
    for e in range(E):
        u, v = src[e].item(), dst[e].item()
        type_u[e], type_v[e], C_geo[e] = D[u], D[v], D
    return type_u, type_v, C_geo

def compute_orc_offline(edge_index, N, supp_u, supp_v, type_u, type_v,
                        C_geo, alpha=0.3, beta=0.4, eps=0.1, n_iter=20, batch=512):
    E = edge_index.shape[1]
    gamma = max(1.0 - alpha - beta, 0.0)
    kappas = []
    device = edge_index.device
    for start in range(0, E, batch):
        end = min(start + batch, E)
        tu = type_u[start:end].to(device); tv = type_v[start:end].to(device)
        C  = C_geo[start:end].to(device)
        t0u=(tu==0).float(); t1u=(tu==1).float(); t2u=(tu==2).float()
        t0v=(tv==0).float(); t1v=(tv==1).float(); t2v=(tv==2).float()
        n1u=t1u.sum(1,keepdim=True).clamp(min=1); n2u=t2u.sum(1,keepdim=True).clamp(min=1)
        n1v=t1v.sum(1,keepdim=True).clamp(min=1); n2v=t2v.sum(1,keepdim=True).clamp(min=1)
        mu = alpha*t0u + (beta/n1u)*t1u + (gamma/n2u)*t2u
        nu = alpha*t0v + (beta/n1v)*t1v + (gamma/n2v)*t2v
        mu = mu / mu.sum(1, keepdim=True).clamp(min=1e-9)
        nu = nu / nu.sum(1, keepdim=True).clamp(min=1e-9)
        W  = sinkhorn_log(mu.log().clamp(min=-30), nu.log().clamp(min=-30),
                          C, eps=eps, n_iter=n_iter)
        kappas.append((1.0 - W).clamp(-1.0, 1.0))
    return torch.cat(kappas)

def _patch_ollivier_ricci_for_windows():
    """GraphRicciCurvature hardcodes mp.get_context('fork') for its edge-curvature
    Pool. 'fork' does not exist on Windows (only 'spawn' does), and the library's
    own comment notes spawn silently breaks its module-level shared state anyway.
    On platforms without 'fork' (Windows), this swaps the parallel Pool.imap_unordered
    call for an in-process serial loop over the exact same per-edge worker function
    and global state, so curvature values are identical to the parallel path — just
    single-core. No-op where 'fork' exists (e.g. the Kaggle T4 GPU env used for the
    experiments reported in the paper), which keeps the original parallel behavior.
    """
    try:
        mp.get_context('fork')
        return
    except ValueError:
        pass
    import GraphRicciCurvature.OllivierRicci as _orc_mod
    if getattr(_orc_mod, '_serial_patch_applied', False):
        return

    def _serial_compute_ricci_curvature_edges(G, weight="weight", edge_list=[],
                                              alpha=0.5, method="OTDSinkhornMix",
                                              base=math.e, exp_power=2, proc=1,
                                              chunksize=None, cache_maxsize=1000000,
                                              shortest_path="all_pairs", nbr_topk=3000):
        if not nx.get_edge_attributes(G, weight):
            for (v1, v2) in G.edges():
                G[v1][v2][weight] = 1.0
        _orc_mod._Gk = _orc_mod.nk.nxadapter.nx2nk(G, weightAttr=weight)
        _orc_mod._alpha, _orc_mod._weight, _orc_mod._method = alpha, weight, method
        _orc_mod._base, _orc_mod._exp_power, _orc_mod._proc = base, exp_power, proc
        _orc_mod._cache_maxsize = cache_maxsize
        _orc_mod._shortest_path, _orc_mod._nbr_topk = shortest_path, nbr_topk
        nx2nk_ndict, nk2nx_ndict = {}, {}
        for idx, n in enumerate(G.nodes()):
            nx2nk_ndict[n] = idx
            nk2nx_ndict[idx] = n
        if shortest_path == "all_pairs":
            _orc_mod._apsp = _orc_mod._get_all_pairs_shortest_path()
        edges = edge_list if edge_list else list(G.edges())
        args = [(nx2nk_ndict[s], nx2nk_ndict[t]) for s, t in edges]
        output = {}
        for a in args:
            rc = _orc_mod._wrap_compute_single_edge(a)
            for k in rc:
                output[(nk2nx_ndict[k[0]], nk2nx_ndict[k[1]])] = rc[k]
        return output

    _orc_mod._compute_ricci_curvature_edges = _serial_compute_ricci_curvature_edges
    _orc_mod._serial_patch_applied = True
    print("  [windows-patch] GraphRicciCurvature: cálculo serial "
          "(mp.get_context('fork') indisponível nesta plataforma)")


def compute_curvature_weights(method, data, device):
    edge_index = data.edge_index
    N = data.num_nodes
    kappa = torch.zeros(edge_index.shape[1], device=device)
    if method == 'forman':
        from GraphRicciCurvature.FormanRicci import FormanRicci
        G = to_networkx(data, to_undirected=True, remove_self_loops=True)
        frc = FormanRicci(G); frc.compute_ricci_curvature()
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0,i].item(), edge_index[1,i].item()
            kappa[i] = 1.0 if u==v else frc.G[u][v].get('formanCurvature', 0.0)
        if kappa.abs().max() > 0: kappa = kappa / kappa.abs().max()
    elif method == 'ollivier':
        import GraphRicciCurvature.OllivierRicci as _orc_mod
        from GraphRicciCurvature.OllivierRicci import OllivierRicci
        _patch_ollivier_ricci_for_windows()
        # GraphRicciCurvature memoizes per-node neighbor distributions in a module-level
        # lru_cache keyed ONLY on the node index, oblivious to which graph is loaded —
        # processing more than one dataset per process silently reuses another graph's
        # cached neighbor data for any overlapping node index (or crashes with an
        # out-of-bounds error when the new graph is smaller). Clearing it before every
        # fresh computation keeps datasets isolated regardless of call order.
        _orc_mod._get_single_node_neighbors_distributions.cache_clear()
        G = to_networkx(data, to_undirected=True, remove_self_loops=True)
        orc = OllivierRicci(G, alpha=0.5, verbose="ERROR"); orc.compute_ricci_curvature()
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0,i].item(), edge_index[1,i].item()
            kappa[i] = 1.0 if u==v else orc.G[u][v].get('ricciCurvature', 0.0)
    elif method == 'a2_overlap':
        A  = to_scipy_sparse_matrix(edge_index, num_nodes=N).tocsr()
        A2 = A.dot(A)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0,i].item(), edge_index[1,i].item()
            kappa[i] = 1.0 if u==v else float(A2[u, v])
        if kappa.max() > 0: kappa = (kappa / kappa.max()) * 2.0 - 1.0
    elif method == 'sinkhorn':
        if N > 5000:
            raise ValueError(
                f"Sinkhorn inviável para N={N} > 5000. "
                "Use 'ollivier' ou 'a2_overlap' para grafos grandes."
            )
        type_u, type_v, C_geo = build_orc_inputs(edge_index.cpu(), N)
        supp = torch.ones(edge_index.shape[1])
        kappa = compute_orc_offline(edge_index, N, supp, supp, type_u, type_v,
                                    C_geo, alpha=0.3, beta=0.4, eps=0.1).to(device)
    else:
        raise ValueError(f"Método '{method}' não reconhecido.")
    return kappa

# ═══════════════════════════════════════════════════════════════
# 2. NCTM
# ═══════════════════════════════════════════════════════════════

def apply_nctm(kappa, nctm_mode, temperatura=1.0):
    if   nctm_mode == 'linear':  return kappa - kappa.min() + 1e-4
    elif nctm_mode == 'sigmoid': return torch.sigmoid(kappa)
    elif nctm_mode == 'sigmatemp': return torch.sigmoid(kappa * temperatura)
    else: raise ValueError(f"NCTM '{nctm_mode}' desconhecido.")

# ═══════════════════════════════════════════════════════════════
# 3. NORMALIZAÇÃO
# ═══════════════════════════════════════════════════════════════

def apply_norm(r, src, dst, N, device, norm_type,
               alpha_val=0.1, kappa=None, tau_alpha=1.0):
    deg_src = torch.zeros(N, device=device).scatter_add_(0, src, r).clamp(min=1e-6)
    deg_dst = torch.zeros(N, device=device).scatter_add_(0, dst, r).clamp(min=1e-6)
    if norm_type == 'src':
        return r / deg_src[src]
    elif norm_type == 'dst':
        return r / deg_dst[dst]
    elif norm_type == 'sym':
        return r / torch.sqrt(deg_src[src] * deg_dst[dst])
    elif norm_type == 'deep_sym':
        deg_2 = torch.zeros(N, device=device).scatter_add_(0, dst, deg_dst[src]).clamp(min=1e-6)
        return r / torch.sqrt(deg_2[src] * deg_2[dst])
    elif norm_type == 'alpha':
        return r / ((deg_src[src] ** alpha_val) * (deg_dst[dst] ** (1.0 - alpha_val)))
    elif norm_type == 'acn':
        if kappa is None: raise ValueError("acn requer kappa.")
        return compute_acn(kappa, r, src, dst, N, device, tau_alpha, deg_src, deg_dst)
    else:
        raise ValueError(f"Norm '{norm_type}' desconhecida.")


def compute_acn(kappa, r, src, dst, N, device, tau_alpha=1.0,
                 deg_src=None, deg_dst=None):
    """
    ACN: Adaptive Curvature Normalization.

    Fórmula da passagem de mensagem com ACN:

        h_i^(l+1) = W_root · h_i^(l) + Σ_{j∈N(i)} r̃_{j→i} · W · h_j^(l)

    onde o peso normalizado por aresta é:

        r̃_{ij} = r'_{ij} / [ (d'_i)^{α_i} · (d'_j)^{1-α_i} ]

    e o expoente adaptativo por nó fonte é:

        α_i = σ(τ_α · κ̄_i)

    A curvatura média local EXCLUI autolaços:

        κ̄_i = Σ_{j∈N(i), j≠i} r'_{ij} · κ_{ij} / Σ_{j∈N(i), j≠i} r'_{ij}

    Motivação da exclusão de autolaços: load_and_prepare adiciona self-loops
    com κ=1.0 hardcoded. Incluí-los inflaria κ̄_i para valores positivos em
    todo grafo, colapsando α_i ≈ 0.5 uniformemente e anulando a adaptatividade
    do ACN — especialmente em grafos heterofílicos onde κ_{ij} << 0.
    """
    # Graus curvatura-ponderados (todos os edges, incluindo self-loops)
    if deg_src is None:
        deg_src = torch.zeros(N, device=device).scatter_add_(0, src, r).clamp(min=1e-6)
    if deg_dst is None:
        deg_dst = torch.zeros(N, device=device).scatter_add_(0, dst, r).clamp(min=1e-6)

    # κ̄_i calculada APENAS sobre arestas reais (sem self-loops)
    no_sl   = src != dst
    r_real     = r[no_sl]
    kappa_real = kappa[no_sl]
    src_r      = src[no_sl]

    num = torch.zeros(N, device=device).scatter_add_(0, src_r, r_real * kappa_real)
    den = torch.zeros(N, device=device).scatter_add_(0, src_r, r_real).clamp(min=1e-6)
    kbar = num / den                              # [N]

    alpha_node = torch.sigmoid(tau_alpha * kbar)  # [N] ∈ (0,1)
    alpha_src  = alpha_node[src]                  # [E] — expoente do nó fonte de cada aresta
    denom = (deg_src[src] ** alpha_src) * (deg_dst[dst] ** (1.0 - alpha_src))
    return r / denom.clamp(min=1e-6)

# ═══════════════════════════════════════════════════════════════
# 3b. DIAGNÓSTICO DO ACN
# ═══════════════════════════════════════════════════════════════

def diagnose_acn(kappa, r_prime, src, dst, N, device, tau_alpha, label=''):
    """
    Imprime estatísticas das variáveis internas do ACN para detectar colapso.

    Monitora três quantidades em sequência:
      1. κ̄_i  — curvatura média local por nó (excluindo self-loops)
      2. α_i  — expoente adaptativo = σ(τ_α · κ̄_i)
      3. r̃_ij — pesos normalizados finais

    Se α_i estiver colapsado em torno de 0.5, o ACN está se comportando como
    normalização simétrica global, indicando que κ̄_i ≈ 0 para todos os nós.
    Causas comuns:
      • self-loops com κ=1.0 ainda poluindo κ̄_i  (verificar no_sl mask)
      • τ_alpha muito pequeno (distribuição de σ muito concentrada)
      • curvatura genuinamente homogênea no grafo
    """
    no_sl      = src != dst
    r_real     = r_prime[no_sl]
    kappa_real = kappa[no_sl]
    src_r      = src[no_sl]

    # κ̄_i (apenas arestas reais)
    num  = torch.zeros(N, device=device).scatter_add_(0, src_r, r_real * kappa_real)
    den  = torch.zeros(N, device=device).scatter_add_(0, src_r, r_real).clamp(min=1e-6)
    kbar = num / den

    # α_i
    alpha_node = torch.sigmoid(tau_alpha * kbar)

    # r̃_ij (pesos finais normalizados pelo ACN)
    deg_src = torch.zeros(N, device=device).scatter_add_(0, src, r_prime).clamp(min=1e-6)
    deg_dst = torch.zeros(N, device=device).scatter_add_(0, dst, r_prime).clamp(min=1e-6)
    alpha_e = alpha_node[src]
    denom_e = (deg_src[src] ** alpha_e) * (deg_dst[dst] ** (1.0 - alpha_e))
    r_tilde = r_prime / denom_e.clamp(min=1e-6)

    def _stats(t, name):
        t = t.float()
        vals = dict(min=t.min().item(), max=t.max().item(),
                    mean=t.mean().item(), std=t.std().item(),
                    p25=t.quantile(0.25).item(), median=t.median().item(),
                    p75=t.quantile(0.75).item())
        spread = vals['max'] - vals['min']
        print(f'    {name:<12}  '
              f'min={vals["min"]:+.4f}  '
              f'p25={vals["p25"]:+.4f}  '
              f'med={vals["median"]:+.4f}  '
              f'p75={vals["p75"]:+.4f}  '
              f'max={vals["max"]:+.4f}  '
              f'std={vals["std"]:.4f}  '
              f'spread={spread:.4f}')
        return vals

    # Text histogram (10 bins) for alpha_i — the most important diagnostic
    def _hist(t, name, n_bins=10):
        t = t.float().cpu()
        lo, hi = t.min().item(), t.max().item()
        if hi - lo < 1e-8:
            print(f'    {name:<12}  [CONSTANTE: {lo:+.6f}  — colapso total]')
            return
        edges = torch.linspace(lo, hi, n_bins + 1)
        counts = torch.histc(t, bins=n_bins, min=lo, max=hi)
        total  = counts.sum().item()
        bar_w  = 30
        print(f'    {name:<12}  histograma ({n_bins} bins, N={int(total)} nós):')
        for i in range(n_bins):
            lo_b = edges[i].item(); hi_b = edges[i+1].item()
            cnt  = int(counts[i].item())
            pct  = cnt / total if total > 0 else 0
            bar  = '█' * int(pct * bar_w)
            print(f'      [{lo_b:+.3f}, {hi_b:+.3f})  {bar:<{bar_w}}  {cnt:5d} ({pct*100:5.1f}%)')

    tag = f' [{label}]' if label else ''
    print(f'\n  ── ACN diagnóstico{tag} (τ_α={tau_alpha}) ──')
    _stats(kbar,        'κ̄_i (nós)')
    _stats(alpha_node,  'α_i  (nós)')
    _stats(r_tilde,     'r̃_ij (arest)')
    _hist(alpha_node,   'α_i  dist.')

    # Colapso warning
    alpha_std = alpha_node.std().item()
    if alpha_std < 0.02:
        print(f'  ⚠  COLAPSO: std(α_i)={alpha_std:.5f} < 0.02  '
              f'→ ACN ≈ normalização simétrica global (sem adaptatividade)')
    elif alpha_std < 0.08:
        print(f'  ⚠  FRACO: std(α_i)={alpha_std:.5f}  '
              f'→ pouca variação entre nós; aumentar τ_α pode ajudar')
    else:
        print(f'  ✓  α_i diversificado (std={alpha_std:.5f})')


# ═══════════════════════════════════════════════════════════════
# 3c. SCATTER α_i × GRAU
# ═══════════════════════════════════════════════════════════════

def plot_acn_scatter(kappa, r_prime, src, dst, N, device,
                      tau_alpha, data, ds_name, label='', out_dir='./figs'):
    """
    Scatter plot: eixo-x = α_i (expoente ACN por nó),
                  eixo-y = grau estrutural do nó (contagem de vizinhos sem self-loops),
                  cor    = classe do nó (data.y).

    Pontos de borda (nós cujos vizinhos têm classe diferente) são marcados com
    um anel preto para destacar sua posição no espaço (α_i, grau).

    Motivação: se α_i ≈ 0.5 para a maioria dos nós, a normalização simétrica
    global seria suficiente. O scatter revela se os nós de borda — aqueles
    críticos para separação de classes — recebem α_i distinto dos nós centrais,
    justificando a adaptatividade do ACN.
    """
    import matplotlib
    matplotlib.use('Agg')          # sem display; salva em arquivo
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    # ── Computar α_i ─────────────────────────────────────────────
    no_sl      = src != dst
    r_real     = r_prime[no_sl];  kappa_real = kappa[no_sl]
    src_r      = src[no_sl];      dst_r      = dst[no_sl]

    num  = torch.zeros(N, device=device).scatter_add_(0, src_r, r_real * kappa_real)
    den  = torch.zeros(N, device=device).scatter_add_(0, src_r, r_real).clamp(min=1e-6)
    kbar = num / den
    alpha_node = torch.sigmoid(tau_alpha * kbar).cpu().numpy()   # [N]

    # ── Grau estrutural (sem self-loops, sem duplicatas direcionadas) ──
    # Conta arestas únicas {i,j} incidentes em cada nó i.
    src_np = src_r.cpu().numpy()
    deg    = np.bincount(src_np, minlength=N).astype(float)       # [N]

    # ── Classes e máscara de borda ────────────────────────────────
    y = data.y.cpu().numpy()                                       # [N]
    # Nó é "de borda" se tem ao menos um vizinho de classe diferente
    is_border = np.zeros(N, dtype=bool)
    for e in range(src_r.shape[0]):
        i, j = src_r[e].item(), dst_r[e].item()
        if y[i] != y[j]:
            is_border[i] = True
            is_border[j] = True

    n_classes  = int(y.max()) + 1
    cmap       = matplotlib.colormaps.get_cmap('tab10').resampled(n_classes)
    colors     = [cmap(int(c)) for c in y]

    # ── Plot: grade 2×2 ───────────────────────────────────────────
    #   linha 0: α_i  vs  grau estrutural  (linear | log)
    #   linha 1: α_i  vs  κ̄_i             (linear | log-módulo)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'{ds_name}  —  ACN scatter  (τ_α={tau_alpha:.2f}  {label})',
        fontsize=13, fontweight='bold'
    )

    def _scatter_panel(ax, x_vals, y_vals, xlabel, ylabel, title,
                       corr_y=None, regression_line=False):
        """Desenha um painel com nós internos + borda e anotação de correlação.

        regression_line : se True, ajusta y = c·x + b e plota a reta com
                          anotação do coeficiente angular c.  Útil para o
                          painel log(1+d'_i) vs α_i, onde a reta implica
                          1+d'_i ~ e^{c·α_i} (relação exponencial).
        """
        ax.scatter(
            x_vals[~is_border], y_vals[~is_border],
            c=[colors[i] for i in np.where(~is_border)[0]],
            alpha=0.45, s=18, linewidths=0, zorder=2,
        )
        ax.scatter(
            x_vals[is_border], y_vals[is_border],
            c=[colors[i] for i in np.where(is_border)[0]],
            alpha=0.85, s=40, linewidths=0.8,
            edgecolors='black', zorder=3,
        )
        ax.axvline(0.5, color='gray', lw=0.8, ls='--', alpha=0.7)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.set_title(title, fontsize=10)
        y_for_corr = corr_y if corr_y is not None else y_vals
        rho = float(np.corrcoef(x_vals, y_for_corr)[0, 1])
        ax.text(0.02, 0.97, f'r = {rho:+.3f}', transform=ax.transAxes,
                va='top', fontsize=9, color='dimgray')

        if regression_line:
            # Ajuste OLS: y_vals = c · x_vals + b
            c, b = np.polyfit(x_vals, y_vals, 1)
            x_line = np.array([x_vals.min(), x_vals.max()])
            ax.plot(x_line, c * x_line + b,
                    color='crimson', lw=1.4, ls='-', alpha=0.8, zorder=4,
                    label=f'regressão: c={c:+.3f}')
            # Anotação do slope no canto superior direito
            ax.text(0.98, 0.97,
                    f'c = {c:+.3f}\n(1+d\') ~ e^{{c·α}}',
                    transform=ax.transAxes,
                    va='top', ha='right', fontsize=8.5,
                    color='crimson',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    alpha_xlabel = r'$\alpha_i$ (expoente ACN)'

    # Linha 0: α_i vs grau
    _scatter_panel(
        axes[0, 0], alpha_node, deg,
        xlabel=alpha_xlabel, ylabel='Grau estrutural',
        title=r'$\alpha_i$ vs grau  (linear)',
    )
    _scatter_panel(
        axes[0, 1], alpha_node, np.log1p(deg),
        xlabel=alpha_xlabel, ylabel='log(1 + grau)',
        title=r'$\alpha_i$ vs grau  (log-grau)',
        corr_y=deg,   # correlação sobre grau bruto
    )

    # ── Grau de curvatura: d'_i = Σ_j r'_ij  (sem self-loops) ───
    # É o denominador do ACN antes de elevar ao α_i.
    deg_curv = torch.zeros(N, device=device).scatter_add_(
        0, src_r, r_real
    ).cpu().numpy()   # [N]

    # Linha 1: α_i vs d'_i (grau ponderado por r'_ij)
    _scatter_panel(
        axes[1, 0], alpha_node, deg_curv,
        xlabel=alpha_xlabel,
        ylabel=r"$d'_i = \sum_j r'_{ij}$  (grau de curvatura)",
        title=r"$\alpha_i$ vs $d'_i$  (linear)",
    )
    _scatter_panel(
        axes[1, 1], alpha_node, np.log1p(deg_curv),
        xlabel=alpha_xlabel,
        ylabel=r"$\log(1 + d'_i)$",
        title=r"$\alpha_i$ vs $d'_i$  (log)  — reta: $\log(1+d') = c\cdot\alpha + b$",
        corr_y=deg_curv,
        regression_line=True,
    )

    # Legenda de classes (lado direito)
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap(c), markersize=7, label=f'classe {c}')
        for c in range(n_classes)
    ]
    handles.append(
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='gray', markeredgecolor='black',
                   markeredgewidth=0.9, markersize=8, label='nó de borda')
    )
    fig.legend(handles=handles, loc='center right',
               bbox_to_anchor=(1.0, 0.5), fontsize=9, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 0.88, 1])

    tag_str  = label.replace(' ', '_').replace('=', '').replace('.', 'p')
    filename = f'{out_dir}/{ds_name}_acn_scatter_{tag_str}.png'
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → figura salva: {filename}')


# ═══════════════════════════════════════════════════════════════
# 3c-bis. DISTRIBUIÇÃO DE CURVATURA E DO EXPOENTE ADAPTATIVO
# (Revisor 1, Major: análise de explicabilidade da curvatura)
# ═══════════════════════════════════════════════════════════════

def plot_curvature_alpha_hist(kappa, r_prime, src, dst, N, device,
                               tau_alpha, ds_name, out_dir='./results/figs'):
    """
    Histograma da curvatura de aresta κ_ij (sem self-loops) e do expoente
    adaptativo por nó α_i = σ(τ_α·κ̄_i), lado a lado.

    Objetivo (Revisor 1): tornar visível a relação entre a geometria do grafo
    e o comportamento do ACN — grafos com curvatura concentrada perto de 0
    produzem α_i concentrado perto de 0.5 (normalização quase simétrica em
    todo o grafo), enquanto grafos com curvatura dispersa/bimodal produzem
    α_i mais espalhado, que é o cenário em que a normalização adaptativa por
    nó tem mais espaço para diferir da melhor normalização fixa global.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    no_sl  = src != dst
    r_real, kappa_real, src_r = r_prime[no_sl], kappa[no_sl], src[no_sl]

    num  = torch.zeros(N, device=device).scatter_add_(0, src_r, r_real * kappa_real)
    den  = torch.zeros(N, device=device).scatter_add_(0, src_r, r_real).clamp(min=1e-6)
    kbar = num / den
    alpha_node = torch.sigmoid(tau_alpha * kbar).cpu().numpy()
    kappa_np   = kappa_real.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(f'{ds_name} — distribuição de curvatura e do expoente adaptativo',
                fontsize=12, fontweight='bold')

    axes[0].hist(kappa_np, bins=40, color='#4C72B0', alpha=0.85, edgecolor='white')
    axes[0].axvline(0.0, color='black', lw=1.0, ls='--', alpha=0.7)
    axes[0].axvline(kappa_np.mean(), color='crimson', lw=1.2, alpha=0.9)
    axes[0].set_xlabel(r'$\kappa_{ij}$ (curvatura de Ollivier-Ricci por aresta)')
    axes[0].set_ylabel('contagem de arestas')
    axes[0].set_title(f'κ: média={kappa_np.mean():+.3f}  std={kappa_np.std():.3f}',
                      fontsize=10)

    axes[1].hist(alpha_node, bins=40, range=(0, 1), color='#DD8452',
                alpha=0.85, edgecolor='white')
    axes[1].axvline(0.5, color='black', lw=1.0, ls='--', alpha=0.7)
    axes[1].axvline(alpha_node.mean(), color='crimson', lw=1.2, alpha=0.9)
    axes[1].set_xlabel(r'$\alpha_i$ (expoente adaptativo por nó)')
    axes[1].set_ylabel('contagem de nós')
    axes[1].set_title(f'α: média={alpha_node.mean():.3f}  std={alpha_node.std():.3f}',
                      fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    filename = f'{out_dir}/{ds_name}_curvature_alpha_hist.png'
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [curvature-hist] κ: média={kappa_np.mean():+.4f} std={kappa_np.std():.4f}  '
          f'| α: média={alpha_node.mean():.4f} std={alpha_node.std():.4f}  '
          f'→ figura salva: {filename}')


def make_combined_curvature_alpha_fig(ds_list, method='ollivier',
                                      out_path='./results/figs/all_datasets_curvature_alpha_hist.png'):
    """
    Single paper figure (Reviewer 1, Major: curvature explainability): 1×len(ds_list)
    grid of the adaptive exponent alpha_i per dataset, side by side. Kappa dispersion
    is already reported in the text (Section on adaptive normalization and curvature
    geometry), so only alpha_i is plotted here to avoid a redundant second row.

    Reuses curvature already cached on disk by compute_curvature_cached; does not
    recompute anything. Run in one isolated process after the 5 datasets have already
    been processed individually (see --explain_only), avoiding GraphRicciCurvature's
    cross-dataset cache issue.

    Palette matches the paper's Figure 1 theme: a diverging green-to-orange
    gradient (vibrant, saturated at the extremes, pale near the center), green
    for target-leaning nodes (alpha_i < 0.5) and orange for source-leaning
    nodes (alpha_i >= 0.5) -- darkest right at 0 and 1, lightest at 0.5.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as _np

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Arial', 'DejaVu Sans']

    DIV_CMAP = LinearSegmentedColormap.from_list(
        'acn_green_orange', ['#1C7A3E', '#F7ECD1', '#E8590C'])
    PANEL_BG = '#EEF2F7'

    device = torch.device('cpu')
    n = len(ds_list)
    fig, axes = plt.subplots(1, n, figsize=(2.75 * n, 3.0))
    if n == 1:
        axes = [axes]

    for j, ds_name in enumerate(ds_list):
        _, data, masks = load_all_splits(ds_name, device)
        N = data.num_nodes
        src, dst = data.edge_index
        data.train_mask, data.val_mask, data.test_mask = masks[0]

        cache_path = f'./cache/kappa_{ds_name}_{method}.pt'
        kappa = torch.load(cache_path, map_location=device)

        no_sl_mask = src != dst
        tmask      = data.train_mask[src] & data.train_mask[dst]
        clean      = no_sl_mask & tmask
        src_c, dst_c = src[clean], dst[clean]
        edge_homophily = ((data.y[src_c] == data.y[dst_c]).float().mean().item()
                          if src_c.numel() > 0 else 0.5)
        tau_alpha = 0.4 if edge_homophily >= 0.5 else 2.0
        r_prime   = apply_nctm(kappa, 'sigmatemp', tau_alpha)

        no_sl  = src != dst
        r_real, kappa_real, src_r = r_prime[no_sl], kappa[no_sl], src[no_sl]
        num  = torch.zeros(N).scatter_add_(0, src_r, r_real * kappa_real)
        den  = torch.zeros(N).scatter_add_(0, src_r, r_real).clamp(min=1e-6)
        alpha_node = torch.sigmoid(tau_alpha * (num / den)).numpy()

        ax = axes[j]
        ax.set_facecolor(PANEL_BG)
        counts, bin_edges = _np.histogram(alpha_node, bins=28, range=(0, 1))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # alpha_i rarely reaches 0/1 in practice, so stretch the colormap to
        # saturate by ~0.2/0.8 instead of the true extremes -- otherwise the
        # visible bars would only ever show the pale middle of the gradient.
        cmap_t = _np.clip((bin_centers - 0.5) / 0.3, -1, 1) * 0.5 + 0.5
        colors = [DIV_CMAP(t) for t in cmap_t]
        ax.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0],
              color=colors, edgecolor='white', linewidth=0.4)
        ax.axvline(0.5, color='#3A3A3A', lw=0.9, ls='--', alpha=0.8)
        ax.axvline(alpha_node.mean(), color='#2A3F5F', lw=1.3)
        ax.set_title(f'{ds_name}\n' + r'$\mu_\alpha$=' + f'{alpha_node.mean():.2f}  '
                    r'$\sigma_\alpha$=' + f'{alpha_node.std():.2f}', fontsize=9.5)
        ax.set_xlabel(r'$\alpha_i$', fontsize=10)
        ax.set_xlim(0, 1)
        if j == 0:
            ax.set_ylabel('node count', fontsize=10)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  -> combined figure saved: {out_path}')


# ═══════════════════════════════════════════════════════════════
# 3d. VISUALIZAÇÃO 2D DAS FEATURES APRENDIDAS
# ═══════════════════════════════════════════════════════════════

def plot_feature_space(model, data, forward_fn, model_name, ds_name, fig_dir):
    """
    Extrai as features da penúltima camada do modelo treinado, reduz para 2D
    via PCA (SVD), e salva um scatter plot colorido por classe.

    Estratégia de extração:
      • Se o modelo tiver atributo 'conv1', registra hook nesse módulo
        (captura a representação oculta antes da camada de classificação).
      • Caso contrário, procura a penúltima camada Linear.

    O hook captura a saída do módulo selecionado durante um forward pass
    completo em modo eval, sem gradientes.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    captured = {}

    # Seleciona o módulo a hookear
    feat_module = None
    if hasattr(model, 'conv1'):
        feat_module = model.conv1
    else:
        linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        if len(linears) >= 2:
            feat_module = linears[-2]

    if feat_module is None:
        return

    def _hook(module, input, output):
        captured['h'] = output.detach().cpu()

    handle = feat_module.register_forward_hook(_hook)
    model.eval()
    with torch.no_grad():
        forward_fn(model, data)
    handle.remove()

    if 'h' not in captured:
        return

    h = captured['h'].float()   # [N, d]
    if h.ndim != 2 or h.shape[1] < 2:
        return

    h_np = (h - h.mean(0)).numpy()   # centrado, [N, d]

    # UMAP 2D (requer umap-learn); fallback para PCA via SVD
    try:
        import warnings, umap as umap_lib
        reducer = umap_lib.UMAP(n_components=2, random_state=42,
                                n_neighbors=15, min_dist=0.1)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning,
                                    module=r'umap\..*')
            h2 = reducer.fit_transform(h_np)   # [N, 2]
        proj_label = 'UMAP'
    except ImportError:
        try:
            _, _, Vt = torch.linalg.svd(torch.from_numpy(h_np), full_matrices=False)
            h2 = (torch.from_numpy(h_np) @ Vt[:2].T).numpy()
            proj_label = 'PCA'
        except Exception:
            return

    y     = data.y.cpu().numpy()
    n_cls = int(y.max()) + 1
    cmap  = matplotlib.colormaps.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in range(n_cls):
        mask = y == c
        ax.scatter(h2[mask, 0], h2[mask, 1],
                   color=cmap(c/10), label=f'class {c}',
                   alpha=0.7, s=20, linewidths=0)
    ax.set_title(f'{ds_name} — {model_name}  [{proj_label}]', fontsize=12)
    ax.set_xlabel(f'{proj_label}-1', fontsize=12)
    ax.set_ylabel(f'{proj_label}-2', fontsize=12)
    ax.legend(fontsize=12, markerscale=1.5, loc='best')
    plt.tight_layout()

    os.makedirs(fig_dir, exist_ok=True)
    safe = (model_name.replace('/', '-').replace(' ', '_')
                      .replace('=', '').replace('.', 'p'))
    path = f'{fig_dir}/{ds_name}_{safe}_feat2d.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  → feat2D: {path}')


# ═══════════════════════════════════════════════════════════════
# 4. MODELOS — BASELINES
# ═══════════════════════════════════════════════════════════════

# ── GCN ───────────────────────────────────────────────────────

class MyGCNConv(MessagePassing):
    def __init__(self, in_c, out_c):
        super().__init__(aggr='add'); self.lin = Linear(in_c, out_c)
    def forward(self, x, ei):
        r, c = ei; d = degree(c, x.size(0), dtype=x.dtype).pow(-0.5)
        d[d == float('inf')] = 0
        return self.propagate(ei, x=self.lin(x), norm=d[r]*d[c])
    def message(self, x_j, norm): return norm.view(-1,1) * x_j

class MyGCN(nn.Module):
    def __init__(self, d_in, d_out, d_hid=64, dropout=0.6):
        super().__init__()
        self.conv1 = MyGCNConv(d_in, d_hid); self.conv2 = MyGCNConv(d_hid, d_out)
        self.p = dropout
    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.relu(self.conv1(F.dropout(x, self.p, self.training), ei))
        return F.log_softmax(self.conv2(F.dropout(x, self.p, self.training), ei), dim=1)

# ── GAT ───────────────────────────────────────────────────────

class GATModel(nn.Module):
    def __init__(self, d_in, d_out, d_hid=64, heads=8, dropout=0.6):
        super().__init__()
        assert d_hid % heads == 0, "d_hid must be divisible by heads"
        self.conv1 = GATConv(d_in,  d_hid // heads, heads=heads,
                              dropout=dropout, concat=True)
        self.conv2 = GATConv(d_hid, d_out, heads=1,
                              dropout=dropout, concat=False)
        self.p = dropout
    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.elu(self.conv1(x, ei))
        x = F.dropout(x, p=self.p, training=self.training)
        return F.log_softmax(self.conv2(x, ei), dim=1)

# ── GraphSAGE ─────────────────────────────────────────────────

class SAGEModel(nn.Module):
    def __init__(self, d_in, d_out, d_hid=64, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(d_in, d_hid)
        self.conv2 = SAGEConv(d_hid, d_out)
        self.p = dropout
    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.relu(self.conv1(F.dropout(x, self.p, self.training), ei))
        return F.log_softmax(self.conv2(F.dropout(x, self.p, self.training), ei), dim=1)

# ── H2GCN (Zhu et al. 2020) ───────────────────────────────────

class H2GCN(nn.Module):
    """
    H2GCN com separação ego-vizinho, vizinhanças 1-hop e 2-hop separadas,
    e skip-connection global sobre todas as camadas.
    Representação final: [h0(hid) || r1(3·hid) || r2(3·hid)] → 7·hid.
    """
    def __init__(self, d_in, d_out, d_hid=64, dropout=0.5):
        super().__init__()
        self.p = dropout
        self.lin_in  = Linear(d_in, d_hid)
        self.W1_ego  = Linear(d_hid,     d_hid, bias=False)
        self.W1_1hop = Linear(d_hid,     d_hid, bias=False)
        self.W1_2hop = Linear(d_hid,     d_hid, bias=False)
        self.W2_ego  = Linear(3 * d_hid, d_hid, bias=False)
        self.W2_1hop = Linear(3 * d_hid, d_hid, bias=False)
        self.W2_2hop = Linear(3 * d_hid, d_hid, bias=False)
        self.classifier = Linear(7 * d_hid, d_out)
        self.norm1 = nn.LayerNorm(3 * d_hid)
        self.norm2 = nn.LayerNorm(3 * d_hid)

    def _agg(self, h, ei, N):
        src, dst = ei[0], ei[1]
        deg = degree(dst, num_nodes=N, dtype=h.dtype).clamp(min=1)
        out = torch.zeros(N, h.shape[1], device=h.device)
        out.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.shape[1]),
                         h[src] / deg[dst].unsqueeze(1))
        return out

    def forward(self, data):
        x, ei = data.x, data.edge_index
        N = x.shape[0]
        x  = F.dropout(x, p=self.p, training=self.training)
        h0 = F.relu(self.lin_in(x))
        h0 = F.dropout(h0, p=self.p * 0.5, training=self.training)

        h1_ego  = self.W1_ego(h0)
        h1_1hop = self.W1_1hop(self._agg(h0, ei, N))
        h1_2hop = self.W1_2hop(self._agg(self._agg(h0, ei, N), ei, N))
        r1 = self.norm1(F.relu(torch.cat([h1_ego, h1_1hop, h1_2hop], dim=-1)))
        r1 = F.dropout(r1, p=self.p, training=self.training)

        h2_ego  = self.W2_ego(r1)
        h2_1hop = self.W2_1hop(self._agg(r1, ei, N))
        h2_2hop = self.W2_2hop(self._agg(self._agg(r1, ei, N), ei, N))
        r2 = self.norm2(F.relu(torch.cat([h2_ego, h2_1hop, h2_2hop], dim=-1)))

        r_all = torch.cat([h0, r1, r2], dim=-1)
        return F.log_softmax(self.classifier(r_all), dim=1)

# ── GPRGNN (Chien et al. 2021) ────────────────────────────────

class GPRGNN(nn.Module):
    """
    Generalized PageRank GNN: aprende pesos polinomiais γ_k sobre K passos
    de propagação com A_hat = D^{-1/2} A D^{-1/2}.
    Inicialização PPR: γ_k = α(1-α)^k, γ_K = (1-α)^K.
    """
    def __init__(self, d_in, d_out, d_hid=64, K=10, alpha=0.1, dropout=0.5):
        super().__init__()
        self.lin1 = Linear(d_in, d_hid)
        self.lin2 = Linear(d_hid, d_out)
        TEMP = alpha * (1 - alpha) ** torch.arange(K + 1, dtype=torch.float)
        TEMP[-1] = (1 - alpha) ** K
        self.gamma = nn.Parameter(TEMP)
        self.K = K; self.p = dropout

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.lin2(x)
        N = x.shape[0]; src, dst = ei
        deg  = degree(dst, num_nodes=N, dtype=x.dtype).clamp(min=1)
        norm = (deg[src] * deg[dst]).sqrt().reciprocal()
        h   = x
        out = self.gamma[0] * x
        for k in range(self.K):
            h_new = torch.zeros_like(h)
            h_new.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.shape[1]),
                               h[src] * norm.unsqueeze(1))
            h = h_new
            out = out + self.gamma[k + 1] * h
        return F.log_softmax(out, dim=1)

# ── APPNP (Klicpera et al. 2019) ─────────────────────────────

class APPNPModel(nn.Module):
    def __init__(self, d_in, d_out, d_hid=64, K=10, alpha=0.1, dropout=0.5):
        super().__init__()
        self.lin1 = Linear(d_in, d_hid)
        self.lin2 = Linear(d_hid, d_out)
        self.prop = APPNPProp(K=K, alpha=alpha)
        self.p = dropout
    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.prop(self.lin2(x), ei)
        return F.log_softmax(x, dim=1)

# ── MixHop (Abu-El-Haija et al. 2019) ────────────────────────

class MixHopModel(nn.Module):
    """
    Mistura representações A^0 h, A^1 h, A^2 h via concatenação.
    Implementado manualmente para evitar dependência de versão do PyG.
    """
    def __init__(self, d_in, d_out, d_hid=21, hops=(0, 1, 2), dropout=0.5):
        super().__init__()
        self.hops = hops
        # Camada 1: para cada hop, projeção independente
        self.W1 = nn.ModuleList([Linear(d_in,          d_hid, bias=False) for _ in hops])
        self.W2 = nn.ModuleList([Linear(d_hid * len(hops), d_out // len(hops) or 1,
                                         bias=False) for _ in hops])
        self.classifier = Linear(d_out // len(hops) * len(hops)
                                  if d_out >= len(hops) else len(hops), d_out)
        self.bn = nn.BatchNorm1d(d_hid * len(hops))
        self.p = dropout

    def _ahat(self, h, ei, N):
        src, dst = ei
        deg  = degree(dst, num_nodes=N, dtype=h.dtype).clamp(min=1)
        norm = (deg[src] * deg[dst]).sqrt().reciprocal()
        out  = torch.zeros_like(h)
        out.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.shape[1]),
                         h[src] * norm.unsqueeze(1))
        return out

    def forward(self, data):
        x, ei = data.x, data.edge_index
        N = x.shape[0]
        x = F.dropout(x, p=self.p, training=self.training)
        hop_feats = []
        for k, W in zip(self.hops, self.W1):
            if k == 0:
                hop_feats.append(W(x))
            else:
                # Apply A_hat k times
                hk = x
                for _ in range(k):
                    hk = self._ahat(hk, ei, N)
                hop_feats.append(W(hk))
        out = F.relu(self.bn(torch.cat(hop_feats, dim=-1)))
        out = F.dropout(out, p=self.p, training=self.training)
        # Layer 2: same multi-hop structure
        hop2 = []
        for k, W in zip(self.hops, self.W2):
            if k == 0:
                hop2.append(W(out))
            else:
                hk = out
                for _ in range(k):
                    hk = self._ahat(hk, ei, N)
                hop2.append(W(hk))
        out2 = torch.cat(hop2, dim=-1)
        return F.log_softmax(self.classifier(out2), dim=1)

# ── CurvGN ───────────────────────────────────────────────────

class CurvGNConv(MessagePassing):
    """
    Curvature Graph Network conv (Ye et al., ICLR 2020), variante CurvGN-n.

    MLP: Linear(1→out_c, bias=False) → LeakyReLU(0.2) → Linear(out_c→out_c)
    Softmax por nó FONTE (edge_index[0]), como no original.
    """
    def __init__(self, in_c, out_c):
        super().__init__(aggr='add')
        self.lin = Linear(in_c, out_c)
        self.w_mlp = nn.Sequential(
            Linear(1, out_c, bias=False), nn.LeakyReLU(0.2), Linear(out_c, out_c)
        )

    def forward(self, x, edge_index, kappa):
        x = self.lin(x)                                       # [N, out_c]
        w = self.w_mlp(kappa.view(-1, 1))                     # [E, out_c]
        w = pyg_softmax(w, edge_index[0], num_nodes=x.size(0))  # per-source, como original
        return self.propagate(edge_index, x=x, w=w)

    def message(self, x_j, w):
        return w * x_j                                        # reponderação por canal


class CurvGN(nn.Module):
    """CurvGN-n (Ye et al., ICLR 2020) — baseline curvatura→MLP de 2 camadas.
    Usa edge_index COM self-loops (A+I) e o kappa alinhado já computado;
    self-loops recebem κ=1.0 e participam do softmax, como no original."""
    def __init__(self, d_in, d_out, d_hid=64, dropout=0.6):
        super().__init__()
        self.conv1 = CurvGNConv(d_in, d_hid)
        self.conv2 = CurvGNConv(d_hid, d_out)
        self.p = dropout

    def forward(self, data, kappa):
        x, ei = data.x, data.edge_index
        x = F.dropout(x, self.p, self.training)
        x = F.elu(self.conv1(x, ei, kappa))
        x = F.dropout(x, self.p, self.training)
        return F.log_softmax(self.conv2(x, ei, kappa), dim=1)

# ── FAGCN ─────────────────────────────────────────────────────

class FAGCNConv(MessagePassing):
    """FAGCN propagation: tanh-gated, degree-normalized messages (no self-residual)."""
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, edge_index, alpha):
        # alpha: [E] pre-computed and dropout'd gate × d_dst × d_src
        return self.propagate(edge_index, x=x, alpha=alpha)

    def message(self, x_j, alpha):
        return alpha.view(-1, 1) * x_j


class FAGCNModel(nn.Module):
    """FAGCN (Bo et al., AAAI 2021): tanh attention × degree normalization;
    fixed pre-loop raw anchor for both residuals."""
    def __init__(self, d_in, d_out, d_hid=64, dropout=0.6, eps=0.1):
        super().__init__()
        self.lin_in = Linear(d_in, d_hid)
        self.att    = Linear(2 * d_hid, 1, bias=False)  # cat([h_dst, h_src])
        self.conv1  = FAGCNConv()
        self.conv2  = FAGCNConv()
        self.out    = Linear(d_hid, d_out)
        self.eps    = eps
        self.p      = dropout

    def _edge_alpha(self, h, ei, deg_norm):
        dst, src = ei[1], ei[0]
        gate = torch.tanh(self.att(torch.cat([h[dst], h[src]], dim=-1))).squeeze(-1)
        return gate * deg_norm[dst] * deg_norm[src]  # [E]

    def forward(self, data):
        x, ei = data.x, data.edge_index
        N = x.size(0)
        deg = degree(ei[1], num_nodes=N, dtype=x.dtype).clamp(min=1)
        deg_norm = deg.pow(-0.5)  # [N]

        x   = F.dropout(x, self.p, self.training)
        raw = F.relu(self.lin_in(x))          # fixed anchor stored before loop

        h     = raw
        alpha = F.dropout(self._edge_alpha(h, ei, deg_norm), self.p, self.training)
        h     = self.conv1(h, ei, alpha)
        h     = self.eps * raw + h            # residual to pre-loop raw

        h     = F.dropout(h, self.p, self.training)
        alpha = F.dropout(self._edge_alpha(h, ei, deg_norm), self.p, self.training)
        h     = self.conv2(h, ei, alpha)
        h     = self.eps * raw + h            # residual to pre-loop raw

        return F.log_softmax(self.out(h), dim=-1)


# ── ACM-GCN ───────────────────────────────────────────────────

class ACMGCNLayer(nn.Module):
    """One ACM-GCN layer: three separate projections with per-node adaptive attention."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_low  = nn.Parameter(torch.empty(d_in, d_out))
        self.W_high = nn.Parameter(torch.empty(d_in, d_out))
        self.W_mlp  = nn.Parameter(torch.empty(d_in, d_out))
        self.a_low  = nn.Parameter(torch.empty(d_out, 1))
        self.a_high = nn.Parameter(torch.empty(d_out, 1))
        self.a_mlp  = nn.Parameter(torch.empty(d_out, 1))
        self.att_vec = nn.Parameter(torch.empty(3, 3))  # 3×3 mixing matrix
        self.reset_parameters()

    def reset_parameters(self):
        stdv_w = self.W_low.size(1) ** -0.5
        for w in (self.W_low, self.W_high, self.W_mlp):
            w.data.uniform_(-stdv_w, stdv_w)
        for a in (self.a_low, self.a_high, self.a_mlp):
            a.data.uniform_(-1.0, 1.0)       # size(1)=1 → stdv=1
        self.att_vec.data.uniform_(-(3 ** -0.5), 3 ** -0.5)

    def _agg_sym(self, h, ei, N):
        src, dst = ei
        deg  = degree(dst, num_nodes=N, dtype=h.dtype).clamp(min=1)
        norm = deg[src].pow(-0.5) * deg[dst].pow(-0.5)
        agg  = torch.zeros_like(h)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, h.shape[1]),
                         h[src] * norm.unsqueeze(1))
        return agg

    def forward(self, x, ei, N):
        xWl, xWh = x @ self.W_low, x @ self.W_high
        low  = F.relu(self._agg_sym(xWl, ei, N))              # relu(A_sym @ xW_l)
        high = F.relu(xWh - self._agg_sym(xWh, ei, N))        # relu((I-A_sym) @ xW_h)
        mlp  = F.relu(x @ self.W_mlp)

        logits = torch.sigmoid(torch.cat([
            low @ self.a_low, high @ self.a_high, mlp @ self.a_mlp
        ], dim=1)) @ self.att_vec / 3.0                        # [N, 3]
        att = F.softmax(logits, dim=1)
        return 3.0 * (att[:, 0:1] * low + att[:, 1:2] * high + att[:, 2:3] * mlp)


class ACMGCNModel(nn.Module):
    """ACM-GCN (Luan et al., NeurIPS 2022): two-layer adaptive channel mixing
    with separate low-pass, high-pass, and identity projections."""
    def __init__(self, d_in, d_out, d_hid=64, dropout=0.5):
        super().__init__()
        self.layer1 = ACMGCNLayer(d_in, d_hid)
        self.layer2 = ACMGCNLayer(d_hid, d_out)
        self.p = dropout

    def forward(self, data):
        x, ei = data.x, data.edge_index
        N = x.size(0)
        x = F.dropout(x, self.p, self.training)
        h = F.relu(self.layer1(x, ei, N))
        h = F.dropout(h, self.p, self.training)
        return F.log_softmax(self.layer2(h, ei, N), dim=-1)


# ── A2GCN ─────────────────────────────────────────────────────

class A2GCNProp(MessagePassing):
    """A2GCN propagation on class logits: Laplacian polynomial filter with per-node sigmoid gates."""
    def __init__(self, K, d_logit):
        super().__init__(aggr='add')
        self.K    = K
        self.temp = nn.Parameter(torch.ones(K))
        self.scores = nn.ParameterList(
            [nn.Parameter(torch.empty(d_logit, 1)) for _ in range(K + 1)])
        self.bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(1))           for _ in range(K + 1)])
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)
        for s in self.scores:
            stdv = 1. / math.sqrt(s.size(1))
            s.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index):
        # L = I − D^{-½}AD^{-½} via get_laplacian(normalization='sym')
        ei_L, norm_L = get_laplacian(edge_index, normalization='sym',
                                     dtype=x.dtype, num_nodes=x.size(0))
        TEMP = torch.tanh(self.temp)                          # [K], per-hop scale ∈ (−1,1)

        gate   = torch.sigmoid(x @ self.scores[0] + self.bias[0])  # [N,1]
        hidden = gate * x
        for k in range(self.K):
            Lx = self.propagate(ei_L, x=x, norm=norm_L)      # L @ x
            x  = x - TEMP[k] * Lx                            # (I − temp_k L) x
            gate   = torch.sigmoid(x @ self.scores[k + 1] + self.bias[k + 1])
            hidden = hidden + gate * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class A2GCNModel(nn.Module):
    """A2GCN (Ai et al., Pattern Recognition 2024): propagation on class logits
    using Laplacian polynomial filter with per-node adaptive gates."""
    def __init__(self, d_in, d_out, d_hid=64, K=2, dropout=0.5, dprate=0.5):
        super().__init__()
        self.lin1 = Linear(d_in, d_hid)
        self.lin2 = Linear(d_hid, d_out)
        self.prop = A2GCNProp(K, d_out)   # propagation on d_out-dim logits
        self.p      = dropout
        self.dprate = dprate

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.lin2(x)                   # [N, d_out] class logits
        if self.dprate == 0.0:
            x = self.prop(x, ei)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop(x, ei)
        return F.log_softmax(x, dim=1)


# ═══════════════════════════════════════════════════════════════
# 5. MODELOS — CGNN e ACN-GCN
# ═══════════════════════════════════════════════════════════════

class CGNNConv(MessagePassing):
    def __init__(self, in_c, out_c, w, p=0.6):
        super().__init__(aggr='add')
        self.lin = Linear(in_c, out_c); self.w = w; self.p = p
    def forward(self, x, ei): return self.propagate(ei, x=self.lin(x))
    def message(self, x_j): return self.w.view(-1, 1) * x_j

class CGNN(nn.Module):
    """Baseline CGNN (Fesser & Günnemann, 2023)."""
    def __init__(self, d_in, d_out, w_mul, d_hid=64, p=0.6):
        super().__init__()
        self.conv1 = CGNNConv(d_in, d_hid, w_mul, p)
        self.conv2 = CGNNConv(d_hid, d_out, w_mul, p)
        self.p = p
    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.relu(self.conv1(F.dropout(x, self.p, self.training), ei))
        return F.log_softmax(self.conv2(F.dropout(x, self.p, self.training), ei), dim=1)

class ACConv(MessagePassing):
    """
    Camada base do ACN-GCN.

    remove_selfloops : remove autolaços antes da propagação (heterofílico)
    root_sep         : W_root · h_v separado de W · h_vizinhos
    """
    def __init__(self, in_c, out_c, p=0.6,
                 remove_selfloops=True, root_sep=True):
        super().__init__(aggr='add')
        self.p = p; self.remove_selfloops = remove_selfloops; self.root_sep = root_sep
        self.lin = Linear(in_c, out_c, bias=True)
        if root_sep:
            self.lin_root = Linear(in_c, out_c, bias=True)

    def forward(self, x, ei, rn):
        ei_p, rn_p = (remove_self_loops(ei, rn) if self.remove_selfloops else (ei, rn))
        out = self.propagate(ei_p, x=self.lin(x), norm=rn_p)
        if self.root_sep:
            out = out + self.lin_root(x)
        return out

    def message(self, x_j, norm): return norm.view(-1,1) * x_j

class ACNGCNModel(nn.Module):
    """
    Adaptive Curvature Normalization Graph Convolution Network (ACN-GCN).
    forward(data, rn1, ei1, rn2, ei2) — interface uniforme com train_and_eval.
    Para escala única: rn1==rn2, ei1==ei2.
    """
    def __init__(self, d_in, d_out, d_hid=64, dropout=0.6,
                 remove_selfloops=True, root_sep=True):
        super().__init__()
        self.conv1 = ACConv(d_in,  d_hid, dropout,
                            remove_selfloops=remove_selfloops, root_sep=root_sep)
        self.conv2 = ACConv(d_hid, d_out, dropout,
                            remove_selfloops=remove_selfloops, root_sep=root_sep)
        self.p = dropout

    def forward(self, data, rn1, ei1, rn2, ei2):
        x = data.x
        x = F.relu(self.conv1(F.dropout(x, self.p, self.training), ei1, rn1))
        return F.log_softmax(self.conv2(F.dropout(x, self.p, self.training), ei2, rn2), dim=1)

# ═══════════════════════════════════════════════════════════════
# 6. TREINAMENTO
# ═══════════════════════════════════════════════════════════════

def train_and_eval(model, data, optimizer, forward_fn=None,
                   epochs=200, patience=50, return_val=False):
    """
    Seleção de modelo por VALIDAÇÃO (sem test-peeking).
    Retorna a acurácia de TESTE no epoch de melhor acurácia de VALIDAÇÃO.
    Se return_val=True, retorna (test_acc, val_acc).
    """
    if forward_fn is None:
        forward_fn = lambda m, d: m(d)
    best_val, best_test, bad = 0.0, 0.0, 0
    for _ in range(epochs):
        model.train(); optimizer.zero_grad()
        out = forward_fn(model, data)
        F.nll_loss(out[data.train_mask], data.y[data.train_mask]).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = forward_fn(model, data).argmax(1)
            val_acc  = (pred[data.val_mask]  == data.y[data.val_mask]).float().mean().item()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        if val_acc > best_val:
            best_val, best_test, bad = val_acc, test_acc, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return (best_test, best_val) if return_val else best_test

# ═══════════════════════════════════════════════════════════════
# 7. HELPERS
# ═══════════════════════════════════════════════════════════════

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_and_prepare(ds_name, device, split_idx=0):
    cfg = DATASETS_CONFIG[ds_name]
    if   cfg['type'] == 'Planetoid': ds = Planetoid(root=f'./data/{ds_name}', name=ds_name)
    elif cfg['type'] == 'WebKB':     ds = WebKB(root=f'./data/{ds_name}', name=ds_name)
    elif cfg['type'] == 'Actor':     ds = Actor(root=f'./data/{ds_name}')
    elif cfg['type'] == 'Wikipedia':
        ds = WikipediaNetwork(root=f'./data/{ds_name}', name=ds_name.lower(),
                              geom_gcn_preprocess=True)
    elif cfg['type'] == 'Hetero':
        ds = HeterophilousGraphDataset(root=f'./data/{ds_name}', name=ds_name)
    else: raise ValueError(cfg['type'])
    data = ds[0]
    if data.train_mask.dim() > 1:                      # [N, n_splits]
        n_splits = data.train_mask.shape[1]
        j = split_idx % n_splits
        data.train_mask = data.train_mask[:, j]
        data.val_mask   = data.val_mask[:, j]
        data.test_mask  = data.test_mask[:, j]
    else:
        n_splits = 1
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)
    return ds, data.to(device), n_splits

def fmt(accs):
    if not accs: return '     —    '
    m, s = np.mean(accs)*100, np.std(accs)*100
    return f'{m:5.1f}±{s:4.1f}'

# ═══════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════

MAX_FIG_NODES = 3000     # gera figuras por-nó só nos grafos pequenos (WebKB)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds_list   = args.datasets if args.datasets else list(DATASETS_CONFIG.keys())
    nctm_opts = [args.nctm]   if args.nctm     else NCTM_ALL
    norm_opts = [args.norm]   if args.norm     else NORM_ALL

    for ds_name in ds_list:
        if ds_name not in DATASETS_CONFIG: continue
        print(f'\n{"═"*70}\n  Dataset: {ds_name}  |  curvatura: {args.method}\n{"═"*70}')

        dataset, data, masks = load_all_splits(ds_name, device)
        n_splits = len(masks)
        N = data.num_nodes
        d_in, d_out = dataset.num_features, dataset.num_classes
        src, dst = data.edge_index
        print(f'  N={N}  E={data.edge_index.shape[1]}  classes={d_out}  splits={n_splits}')

        kappa = compute_curvature_cached(args.method, data, device, ds_name)

        # ── Regime inferido UMA vez no split 0 (sem vazamento) ──
        data.train_mask, data.val_mask, data.test_mask = masks[0]
        no_sl_mask = src != dst
        tmask      = data.train_mask[src] & data.train_mask[dst]
        clean      = no_sl_mask & tmask
        src_c, dst_c = src[clean], dst[clean]
        edge_homophily = ((data.y[src_c] == data.y[dst_c]).float().mean().item()
                          if src_c.numel() > 0 else 0.5)
        is_homo = edge_homophily >= 0.5
        lr, wd        = (0.01, 5e-4) if is_homo else (0.05, 5e-5)
        tau_sigmatemp = 0.4 if is_homo else 2.0
        tau_alpha     = 0.4 if is_homo else 2.0
        rm_sl         = not is_homo
        print(f'  homofilia(treino)={edge_homophily:.3f} → '
              f'{"homo" if is_homo else "hetero"}  (τ={tau_sigmatemp}  rm_sl={rm_sl})')

        # ── Diagnóstico ACN (split 0; figuras só p/ grafos pequenos) ──
        _r_diag = apply_nctm(kappa, 'sigmatemp', tau_sigmatemp)
        diagnose_acn(kappa, _r_diag, src, dst, N, device,
                     tau_alpha=tau_alpha, label=f'τ={tau_sigmatemp}')
        if N <= MAX_FIG_NODES:
            plot_acn_scatter(kappa, _r_diag, src, dst, N, device, tau_alpha,
                             data, ds_name, label=f'tau{tau_sigmatemp}')
            plot_curvature_alpha_hist(kappa, _r_diag, src, dst, N, device,
                                      tau_alpha, ds_name)
        # dispersão de curvatura — alimenta tabela do paper
        _r_ns   = _r_diag[no_sl_mask]
        _k_ns   = kappa[no_sl_mask]
        _src_ns = src[no_sl_mask]
        _num_a  = torch.zeros(N, device=device).scatter_add_(0, _src_ns, _r_ns * _k_ns)
        _den_a  = torch.zeros(N, device=device).scatter_add_(0, _src_ns, _r_ns).clamp(min=1e-6)
        _alpha_i = torch.sigmoid(tau_alpha * (_num_a / _den_a))
        print(f'  [dispersão] std(κ)={_k_ns.std().item():.4f}'
              f'  std(α_i)={_alpha_i.std().item():.4f}')
        del _r_ns, _k_ns, _src_ns, _num_a, _den_a, _alpha_i
        del _r_diag

        if getattr(args, 'explain_only', False):
            continue   # só curvatura + figuras de explicabilidade, sem treinar nada

        make_opt_lr  = lambda m: torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
        make_opt_gcn = lambda m: torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)

        baseline_accs   = defaultdict(list)
        norm_sweep_accs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        ablation_accs   = defaultdict(lambda: defaultdict(list))
        ablation_val    = defaultdict(lambda: defaultdict(list))
        fig_dir = f'./results/{ds_name}/features'

        # ════════════ BASELINES (split × seed) ════════════
        print(f'\n  ── Baselines ({n_splits}×{len(SEEDS)} runs) ──')
        for split_idx in range(n_splits):
            data.train_mask, data.val_mask, data.test_mask = masks[split_idx]
            for seed in SEEDS:
                set_seed(seed)
                # treinar e avaliar (otimizador deve referir-se ao MESMO modelo):
                for key, build in [
                    ('GCN',       lambda: (MyGCN(d_in, d_out).to(device),       make_opt_gcn)),
                    ('GAT',       lambda: (GATModel(d_in, d_out).to(device),    make_opt_lr)),
                    ('GraphSAGE', lambda: (SAGEModel(d_in, d_out).to(device),   make_opt_lr)),
                    ('H2GCN',     lambda: (H2GCN(d_in, d_out).to(device),       make_opt_lr)),
                    ('GPRGNN',    lambda: (GPRGNN(d_in, d_out, K=args.prop_steps,
                                                  alpha=args.prop_alpha).to(device), make_opt_lr)),
                    ('APPNP',     lambda: (APPNPModel(d_in, d_out, K=args.prop_steps,
                                                      alpha=args.prop_alpha).to(device), make_opt_lr)),
                    ('MixHop',    lambda: (MixHopModel(d_in, d_out).to(device),  make_opt_lr)),
                    ('FAGCN',     lambda: (FAGCNModel(d_in, d_out).to(device),   make_opt_lr)),
                    ('ACM-GCN',   lambda: (ACMGCNModel(d_in, d_out).to(device),  make_opt_lr)),
                    ('A2GCN',     lambda: (A2GCNModel(d_in, d_out, K=10).to(device), make_opt_lr)),
                ]:
                    m, opt_fn = build()
                    baseline_accs[key].append(
                        train_and_eval(m, data, opt_fn(m), epochs=args.epochs))
                # CurvGN (kappa como argumento extra)
                m = CurvGN(d_in, d_out).to(device)
                baseline_accs['CurvGN'].append(train_and_eval(
                    m, data, make_opt_lr(m),
                    forward_fn=lambda m, d, _k=kappa: m(d, _k), epochs=args.epochs))
        for key in BASELINE_KEYS:
            print(f'    {key:<12}: {fmt(baseline_accs[key])}')

        # ════════════ BLOCO 1: CGNN × nctm × norm ════════════
        print(f'\n  ── Bloco 1: norm sweep (CGNN) ──')
        for nctm in nctm_opts:
            for norm in norm_opts:
                tau_n  = tau_sigmatemp if nctm == 'sigmatemp' else 1.0
                r_nctm = apply_nctm(kappa, nctm, tau_n)
                r_norm = apply_norm(r_nctm, src, dst, N, device, norm,
                                    alpha_val=args.alpha_val, kappa=kappa, tau_alpha=tau_alpha)
                last_m = None
                for split_idx in range(n_splits):
                    data.train_mask, data.val_mask, data.test_mask = masks[split_idx]
                    for seed in SEEDS:
                        set_seed(seed)
                        m = CGNN(d_in, d_out, r_norm).to(device)
                        norm_sweep_accs[nctm][norm]['CGNN'].append(
                            train_and_eval(m, data, make_opt_lr(m), epochs=args.epochs))
                        last_m = m
                    if split_idx == 0 and N <= MAX_FIG_NODES:
                        plot_feature_space(last_m, data, lambda m, d: m(d),
                            f'CGNN_nctm={nctm}_norm={norm}', ds_name, fig_dir)
                print(f'    nctm={nctm:<10} norm={norm:<8} '
                      f"CGNN={fmt(norm_sweep_accs[nctm][norm]['CGNN'])}")

        ns_dir = f'./results/{ds_name}/norm_sweep'; os.makedirs(ns_dir, exist_ok=True)
        with open(f'{ns_dir}/all_norms.json', 'w') as f:
            json.dump({'baselines': dict(baseline_accs),
                       'norm_sweep': {nc: {no: dict(md) for no, md in nm.items()}
                                      for nc, nm in norm_sweep_accs.items()}}, f, indent=2)

        # ════════════ BLOCO 2: ablação ACN-GCN (sigmatemp) ════════════
        print(f'\n  ── Bloco 2: ablação ACN-GCN ──')
        r_base = apply_nctm(kappa, 'sigmatemp', tau_sigmatemp)
        ei = data.edge_index
        for norm in norm_opts:
            r_norm = apply_norm(r_base, src, dst, N, device, norm,
                                alpha_val=args.alpha_val, kappa=kappa, tau_alpha=tau_alpha)
            fwd = lambda m, d, rn=r_norm, _ei=ei: m(d, rn, _ei, rn, _ei)
            m_nr0 = m_rs0 = None
            for split_idx in range(n_splits):
                data.train_mask, data.val_mask, data.test_mask = masks[split_idx]
                for seed in SEEDS:
                    set_seed(seed)
                    m_nr = ACNGCNModel(d_in, d_out, remove_selfloops=rm_sl,
                                       root_sep=False).to(device)
                    _t_nr, _v_nr = train_and_eval(m_nr, data, make_opt_lr(m_nr),
                                       forward_fn=fwd, epochs=args.epochs, return_val=True)
                    ablation_accs[norm]['ACN-GCN-noroot'].append(_t_nr)
                    ablation_val[norm]['ACN-GCN-noroot'].append(_v_nr)
                    m_rs = ACNGCNModel(d_in, d_out, remove_selfloops=rm_sl,
                                       root_sep=True).to(device)
                    _t_rs, _v_rs = train_and_eval(m_rs, data, make_opt_lr(m_rs),
                                       forward_fn=fwd, epochs=args.epochs, return_val=True)
                    ablation_accs[norm]['ACN-GCN'].append(_t_rs)
                    ablation_val[norm]['ACN-GCN'].append(_v_rs)
                if split_idx == 0:
                    m_nr0, m_rs0 = m_nr, m_rs
            if N <= MAX_FIG_NODES and m_rs0 is not None:
                plot_feature_space(m_nr0, data, fwd, f'ACN-GCN-noroot_norm={norm}',
                                   ds_name, fig_dir)
                plot_feature_space(m_rs0, data, fwd, f'ACN-GCN_norm={norm}',
                                   ds_name, fig_dir)
            print(f"    norm={norm:<8} "
                  f"noroot={fmt(ablation_accs[norm]['ACN-GCN-noroot'])}  "
                  f"ACN-GCN={fmt(ablation_accs[norm]['ACN-GCN'])}")

        abl_dir = f'./results/{ds_name}/ablation'; os.makedirs(abl_dir, exist_ok=True)
        with open(f'{abl_dir}/chain_sym.json', 'w') as f:
            json.dump({no: dict(md) for no, md in ablation_accs.items()}, f, indent=2)

        sel_norm = select_norm_by_val(ablation_val)
        _print_significance(baseline_accs, ablation_accs, sel_norm)
        _print_results(ds_name, baseline_accs, norm_sweep_accs, ablation_accs,
                       nctm_opts, norm_opts)


def _print_results(ds_name, baseline_accs, norm_sweep_accs, ablation_accs,
                   nctm_opts, norm_opts):
    W = 16
    print(f'\n{"═"*72}')
    print(f'  RESULTADOS — {ds_name}')
    print(f'{"═"*72}')

    # ── Baselines ─────────────────────────────────────────────────────
    print(f'\n  {"Baselines":─<70}')
    for key in BASELINE_KEYS:
        accs = baseline_accs.get(key, [])
        print(f'  {key:<14}: {fmt(accs)}')

    # ── Bloco 1: CGNN norm sweep ───────────────────────────────────────
    print(f'\n  {"Bloco 1 — CGNN (nctm × norm)":─<70}')
    hdr = f"  {'nctm/norm':<18}" + ''.join(f'{k:>{W}}' for k in NORM_SWEEP_KEYS)
    print(hdr)
    print(f'  {"─"*60}')
    prev_nctm = None
    for nctm in nctm_opts:
        if nctm != prev_nctm and prev_nctm is not None:
            print(f'  {"─"*60}')
        prev_nctm = nctm
        for norm in norm_opts:
            row = f'  {nctm+"/"+norm:<18}'
            for key in NORM_SWEEP_KEYS:
                row += f'{fmt(norm_sweep_accs[nctm][norm].get(key, [])):>{W}}'
            print(row)

    # ── Bloco 2: ACN-GCN cadeia de ablação ────────────────────────────
    print(f'\n  {"Bloco 2 — ACN-GCN ablação (nctm=sigmatemp)":─<70}')
    CHAIN_KEYS = ['CGNN'] + ABLATION_KEYS   # CGNN → noroot → ACN-GCN
    hdr = f"  {'norm':<18}" + ''.join(f'{k:>{W}}' for k in CHAIN_KEYS)
    print(hdr)
    print(f'  {"─"*60}')
    for norm in norm_opts:
        row = f'  {norm:<18}'
        row += f'{fmt(norm_sweep_accs.get("sigmatemp", {}).get(norm, {}).get("CGNN", [])):>{W}}'
        for key in ABLATION_KEYS:
            row += f'{fmt(ablation_accs[norm].get(key, [])):>{W}}'
        print(row)

    # ── Cadeia focada (norm=acn): responde a pergunta de pesquisa ──────
    print(f'\n  {"Cadeia focada — norm=acn (cada linha = uma contribuição)":─<70}')
    _chain = [
        ('CGNN + sym (fixo)',     norm_sweep_accs.get('sigmatemp',{}).get('sym',{}).get('CGNN',[]),
         'curvatura + norm fixa'),
        ('CGNN + acn',            norm_sweep_accs.get('sigmatemp',{}).get('acn',{}).get('CGNN',[]),
         '+ normalização adaptativa (α_i)'),
        ('ACN-GCN-noroot + acn',  ablation_accs.get('acn',{}).get('ACN-GCN-noroot',[]),
         '+ remoção self-loops (hetero)'),
        ('ACN-GCN + acn',         ablation_accs.get('acn',{}).get('ACN-GCN',[]),
         '+ W_root separado'),
    ]
    print(f"  {'Modelo':<28}  {'Acc':>12}  Contribuição")
    print(f'  {"─"*68}')
    for label, accs, contrib in _chain:
        print(f'  {label:<28}  {fmt(accs):>12}  {contrib}')

    # ── Melhor por modelo (máximo sobre sweep) ─────────────────────────
    print(f'\n  {"Best (max sobre sweep)":─<70}')
    for key in NORM_SWEEP_KEYS:
        all_runs = [acc for nctm in nctm_opts
                        for norm in norm_opts
                        for acc in norm_sweep_accs[nctm][norm].get(key, [])]
        print(f'  {key:<22}: {fmt(all_runs)}  (n={len(all_runs)})')
    for key in ABLATION_KEYS:
        all_runs = [acc for norm in norm_opts
                        for acc in ablation_accs[norm].get(key, [])]
        print(f'  {key:<22}: {fmt(all_runs)}  (n={len(all_runs)})')
    print(f'{"═"*72}')


# ═══════════════════════════════════════════════════════════════
# 9. BENCHMARK DE CUSTO COMPUTACIONAL
# (Revisor 4, Major: trade-off de custo/complexidade do ACN)
# ═══════════════════════════════════════════════════════════════

def benchmark_computational_cost(ds_list, method='ollivier', n_time_epochs=20):
    """
    Mede o overhead real do ACN frente à normalização fixa 'sym', para
    responder ao pedido do Revisor 4 de discutir explicitamente o trade-off
    de custo, já que o ACN não bate a melhor normalização fixa em nenhum
    dataset:

      (a) pré-processamento: curvatura de Ollivier-Ricci computada do zero
          (sem cache — o objetivo é medir o custo de tê-la que calcular, não
          o custo de lê-la de um cache já quente) + apply_norm('sym') vs
          apply_norm('acn');
      (b) custo por época: ACNGCNModel treinando sob o mesmo r_norm fixo,
          variando apenas se ele veio de 'sym' ou 'acn' — a arquitetura de
          propagação (ACConv) é idêntica nos dois casos.

    Resultados salvos em ./results/timing/{ds_name}_timing.json e
    ./results/timing/all_timing.json.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('./results/timing', exist_ok=True)

    # Aquecimento: a primeira chamada a compute_curvature_weights('ollivier', ...) em
    # todo o processo paga um custo único de inicialização do backend (networkit/POT),
    # que não tem relação com o tamanho do grafo real e infla injustamente o primeiro
    # dataset da lista. Paga-se esse custo aqui, numa vez descartável, num grafo minúsculo.
    _warm_G = nx.path_graph(6)
    _warm_ei = torch.tensor(list(_warm_G.edges())).t().contiguous()
    _warm_ei = torch.cat([_warm_ei, _warm_ei.flip(0)], dim=1).to(device)
    _warm_data = _PygData(edge_index=_warm_ei, num_nodes=6)
    _warm_kappa = compute_curvature_weights(method, _warm_data, device)
    # também aquece os kernels CUDA de scatter_add usados por apply_norm (sym e acn),
    # separados do aquecimento acima, que só cobre o backend de curvatura.
    apply_norm(_warm_kappa, _warm_ei[0], _warm_ei[1], 6, device, 'sym')
    apply_norm(_warm_kappa, _warm_ei[0], _warm_ei[1], 6, device, 'acn',
              kappa=_warm_kappa, tau_alpha=1.0)
    if device.type == 'cuda': torch.cuda.synchronize()
    del _warm_G, _warm_ei, _warm_data, _warm_kappa

    all_results = {}
    for ds_name in ds_list:
        print(f'\n{"─"*60}\n  [timing] {ds_name}  (device={device})\n{"─"*60}')
        dataset, data, masks = load_all_splits(ds_name, device)
        data.train_mask, data.val_mask, data.test_mask = masks[0]
        N = data.num_nodes
        d_in, d_out = dataset.num_features, dataset.num_classes
        src, dst = data.edge_index

        # (a) curvatura computada do zero, sem cache
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        kappa = compute_curvature_weights(method, data, device)
        if device.type == 'cuda': torch.cuda.synchronize()
        t_curvature = time.time() - t0

        no_sl_mask   = src != dst
        tmask        = data.train_mask[src] & data.train_mask[dst]
        clean        = no_sl_mask & tmask
        src_c, dst_c = src[clean], dst[clean]
        edge_homophily = ((data.y[src_c] == data.y[dst_c]).float().mean().item()
                          if src_c.numel() > 0 else 0.5)
        is_homo = edge_homophily >= 0.5
        tau     = 0.4 if is_homo else 2.0
        rm_sl   = not is_homo
        r_base  = apply_nctm(kappa, 'sigmatemp', tau)

        # (b) pré-processamento da normalização: sym (barata) vs acn (usa kappa)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        r_sym = apply_norm(r_base, src, dst, N, device, 'sym')
        if device.type == 'cuda': torch.cuda.synchronize()
        t_norm_sym = time.time() - t0

        t0 = time.time()
        r_acn = apply_norm(r_base, src, dst, N, device, 'acn', kappa=kappa, tau_alpha=tau)
        if device.type == 'cuda': torch.cuda.synchronize()
        t_norm_acn = time.time() - t0

        # (c) custo por época: mesma arquitetura, só muda o tensor r_norm de entrada
        ei = data.edge_index
        def _time_epochs(r_norm, n=n_time_epochs, n_warmup=10):
            set_seed(42)
            m   = ACNGCNModel(d_in, d_out, remove_selfloops=rm_sl, root_sep=True).to(device)
            opt = torch.optim.Adam(m.parameters(), lr=0.05, weight_decay=5e-5)
            fwd = lambda mm, dd: mm(dd, r_norm, ei, r_norm, ei)
            # Aquecimento: descarta as primeiras épocas para não confundir a medição
            # com o custo único de alocação de memória / autotuning de kernels CUDA,
            # que de outro modo penalizaria injustamente a QUALQUER variante que seja
            # cronometrada primeiro (sym é sempre chamado antes de acn abaixo).
            for _ in range(n_warmup):
                m.train(); opt.zero_grad()
                out = fwd(m, data)
                F.nll_loss(out[data.train_mask], data.y[data.train_mask]).backward()
                opt.step()
            if device.type == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                m.train(); opt.zero_grad()
                out = fwd(m, data)
                F.nll_loss(out[data.train_mask], data.y[data.train_mask]).backward()
                opt.step()
            if device.type == 'cuda': torch.cuda.synchronize()
            return (time.time() - t0) / n

        t_epoch_sym = _time_epochs(r_sym)
        t_epoch_acn = _time_epochs(r_acn)

        res = dict(N=N, E=int(data.edge_index.shape[1]), device=str(device),
                   curvature_s=t_curvature,
                   norm_sym_ms=t_norm_sym * 1000, norm_acn_ms=t_norm_acn * 1000,
                   epoch_sym_ms=t_epoch_sym * 1000, epoch_acn_ms=t_epoch_acn * 1000)
        all_results[ds_name] = res
        print(f'  curvatura={t_curvature:.3f}s  |  '
              f'norm(sym)={t_norm_sym*1000:.3f}ms  norm(acn)={t_norm_acn*1000:.3f}ms  |  '
              f'época(sym)={t_epoch_sym*1000:.3f}ms  época(acn)={t_epoch_acn*1000:.3f}ms')
        with open(f'./results/timing/{ds_name}_timing.json', 'w') as f:
            json.dump(res, f, indent=2)
    with open('./results/timing/all_timing.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    return all_results


# ═══════════════════════════════════════════════════════════════
# ARGPARSE
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='ACN-GCN — Ablação + Baselines')

    p.add_argument('--method', default='ollivier',
                   choices=['forman', 'ollivier', 'a2_overlap', 'sinkhorn'])
    p.add_argument('--datasets', nargs='+', default=None,
                   choices=list(DATASETS_CONFIG.keys()))
    p.add_argument('--nctm', default=None, choices=['linear', 'sigmoid', 'sigmatemp'])
    p.add_argument('--norm', default=None,
                   choices=['src', 'dst', 'sym', 'deep_sym', 'alpha', 'acn'])
    p.add_argument('--alpha_val',   type=float, default=0.1)
    p.add_argument('--epochs',      type=int,   default=200)
    p.add_argument('--prop_steps',  type=int,   default=10,
                   help='Passos de propagação para GPRGNN e APPNP (K)')
    p.add_argument('--prop_alpha',  type=float, default=0.1,
                   help='Coeficiente de teleporte para GPRGNN e APPNP (α)')
    p.add_argument('--benchmark_cost', action='store_true',
                   help='Roda só o benchmark de custo computacional (Revisor 4) e sai, '
                        'sem rodar a suíte completa de experimentos.')
    p.add_argument('--benchmark_epochs', type=int, default=20,
                   help='Número de épocas cronometradas por variante no benchmark de custo.')
    p.add_argument('--explain_only', action='store_true',
                   help='Gera só curvatura + figuras de explicabilidade (histograma κ/α, '
                        'scatter, diagnóstico) por dataset e sai, sem treinar nenhum modelo.')
    p.add_argument('--make_summary_fig', action='store_true',
                   help='Monta a figura combinada κ/α (todos os datasets lado a lado) a '
                        'partir da curvatura já cacheada em disco e sai.')

    args, _ = p.parse_known_args()
    if args.make_summary_fig:
        ds_list = args.datasets if args.datasets else list(DATASETS_CONFIG.keys())
        make_combined_curvature_alpha_fig(ds_list, method=args.method)
    elif args.benchmark_cost:
        ds_list = args.datasets if args.datasets else list(DATASETS_CONFIG.keys())
        benchmark_computational_cost(ds_list, method=args.method,
                                     n_time_epochs=args.benchmark_epochs)
    else:
        main(args)
