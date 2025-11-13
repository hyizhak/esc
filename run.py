import argparse
from collections import OrderedDict
import datetime
import json
import math
import random
from os import listdir, makedirs
from os.path import basename, isdir, isfile, join
from typing import Dict, List, Optional

from sklearn.model_selection import KFold

import dgl
from dgl.dataloading import GraphDataLoader
from dgl.nn import RelGraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from file_utils import read_data


def create_vocab(m2_dir, data_dir, source_name, target_name):
    """
    Generate a vocabulary of edit types
    """
    src_path = join(data_dir, source_name)
    target_path = join(data_dir, target_name)
    target_m2 = read_data(src_path, target_path, m2_dir)
    edit_types = set([])
    for instance in target_m2:
        edit_types |= set([e[2] for e in instance['edits']])
    edit_types = sorted(list(edit_types))
    edit_types = {e: i for i, e in enumerate(edit_types)}

    return edit_types


class FrozenContextualEncoder:
    """
    Utility wrapper around a frozen Transformer encoder (e.g., RoBERTa) that
    produces per-token contextual embeddings for whitespace-tokenized sentences.
    """

    def __init__(self, model_name: str, layer: int = -1, max_length: int = 256, device: str = "cpu"):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for --contextual_encoder usage. "
                "Please install it via pip install transformers."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if hasattr(self.tokenizer, "add_prefix_space"):
            try:
                self.tokenizer.add_prefix_space = True
            except AttributeError:
                pass
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(device)

        self.hidden_size = self.model.config.hidden_size
        self.layer = layer
        self.max_tokens = max_length
        self.device = device
        self.use_hidden_states = layer not in (-1, None)

    def _select_hidden(self, outputs):
        if self.use_hidden_states:
            hidden_states = outputs.hidden_states
            layer_index = self.layer
            if layer_index < 0:
                layer_index = len(hidden_states) + layer_index
            layer_index = max(0, min(layer_index, len(hidden_states) - 1))
            return hidden_states[layer_index][0]
        return outputs.last_hidden_state[0]

    def encode(self, tokens: List[str]) -> torch.Tensor:
        if not tokens:
            return torch.zeros((0, self.hidden_size), dtype=torch.float)

        token_embeddings = torch.zeros((len(tokens), self.hidden_size), dtype=torch.float)
        start = 0
        while start < len(tokens):
            end = min(len(tokens), start + self.max_tokens)
            segment = tokens[start:end]
            tokenized = self.tokenizer(
                segment,
                is_split_into_words=True,
                return_tensors="pt",
                padding=False,
                truncation=False,
                add_special_tokens=True,
            )
            word_ids = tokenized.word_ids()
            encoding = {k: v.to(self.device) for k, v in tokenized.items()}
            with torch.no_grad():
                outputs = self.model(**encoding, output_hidden_states=self.use_hidden_states)
            hidden = self._select_hidden(outputs).cpu()

            # Aggregate subword pieces per original token.
            sums: Dict[int, torch.Tensor] = {}
            counts: Dict[int, int] = {}
            for idx, wid in enumerate(word_ids):
                if wid is None:
                    continue
                sums.setdefault(wid, torch.zeros(self.hidden_size, dtype=torch.float))
                sums[wid] += hidden[idx]
                counts[wid] = counts.get(wid, 0) + 1
            for local_idx, vec_sum in sums.items():
                token_embeddings[start + local_idx] = vec_sum / max(counts[local_idx], 1)
            start = end
        return token_embeddings


class M2Dataset(Dataset):
    def __init__(self, m2_dir, data_dir, source_name, target_name, vocab,
                    filter_idx=None, test=False, upsample=None,
                    max_hypotheses: Optional[int] = None,
                    allowed_hypotheses: Optional[List[str]] = None,
                    contextualizer: Optional[FrozenContextualEncoder] = None,
                    use_global_node: bool = True):
        """
        Dataset that groups edits per sentence and builds DGL graphs for
        graph-based reasoning over candidate edits.
        """
        self.test = test
        if not isdir(m2_dir):
            makedirs(m2_dir)
        
        src_path = join(data_dir, source_name)
        
        if not test and target_name is not None:
            target_path = join(data_dir, target_name)
            target_m2 = read_data(src_path, target_path, m2_dir, filter_idx=filter_idx)
        else:
            target_m2 = None

        self.edit_types = vocab['edit_types']
        self.hyp_list = self._select_hypotheses(vocab['hyp_list'], max_hypotheses, allowed_hypotheses)
        
        # MODIFIED: Store counts for creating multi-hot vectors
        self.num_edit_types = len(self.edit_types)
        self.num_hyps = len(self.hyp_list)
        
        self.extra_feature_size = 5
        
        self.relation_types = {
            'same_hypothesis': 0,
            'same_span': 1,
            'overlap': 2,
            'insertion': 3,
            'adjacent': 4,    
            'self_loop': 5,   
            'global': 6,      
        }
        self.num_relations = len(self.relation_types)
        self.contextualizer = contextualizer
        self.context_cache = {} if contextualizer is not None else None
        self.context_feature_size = contextualizer.hidden_size if contextualizer is not None else 0
        self.use_global_node = use_global_node

        self.graphs = []
        self.graph_edit_keys = []
        if test:
            self.all_edits = []

        data = []
        for file_name in self.hyp_list:
            print('Loading {}...'.format(file_name))
            file_path = join(data_dir, file_name)
            hyp_data = read_data(src_path, file_path, m2_dir, target_m2, filter_idx)
            data.append(hyp_data)
        
        doc_lens = [len(d) for d in data]
        assert min(doc_lens) == max(doc_lens), "M2 lengths are different!"

        if upsample is not None:
            raise NotImplementedError("Upsampling is not supported in the GNN pipeline.")

        self.transform(data, self.edit_types, test)
       
        if not test:
            self.label_counts()
            print('Label distribution: ', self.label_count)

    def _select_hypotheses(self, hyp_list, max_hypotheses, allowed_hypotheses):
        selected = hyp_list
        if allowed_hypotheses:
            allowed_set = set(allowed_hypotheses)
            filtered = [h for h in hyp_list if h in allowed_set]
            if filtered:
                selected = filtered
        if max_hypotheses is not None and max_hypotheses > 0:
            selected = selected[:max_hypotheses]
        return selected

    # MODIFIED: _build_graph signature changed to accept multi-hot and contextual features
    def _build_graph(self, scalar_features, type_features, hyp_features, context_features,
                     labels, edit_keys, hyp_sets, sentence_len):
        num_nodes = len(scalar_features)
        span_features = [[start, end] for start, end, _ in edit_keys]
        if num_nodes == 0:
            g = dgl.graph(([], []), num_nodes=0)
            # MODIFIED: Initialize multi-hot feature tensors
            g.ndata['scalar_feat'] = torch.zeros((0, self.extra_feature_size), dtype=torch.float)
            g.ndata['type_feat'] = torch.zeros((0, self.num_edit_types), dtype=torch.float)
            g.ndata['hyp_feat'] = torch.zeros((0, self.num_hyps), dtype=torch.float)
            if self.context_feature_size > 0:
                g.ndata['context_feat'] = torch.zeros((0, self.context_feature_size), dtype=torch.float)
            g.ndata['span'] = torch.zeros((0, 2), dtype=torch.long)
            g.ndata['label'] = torch.zeros(0, dtype=torch.float)
            g.ndata['mask'] = torch.zeros(0, dtype=torch.bool)
            g.edata['rel_type'] = torch.zeros(0, dtype=torch.int64)
            return g

        src, dst, rel_types = self._build_edges(edit_keys, hyp_sets)
        total_nodes = num_nodes
        if self.use_global_node:
            global_idx = total_nodes
            total_nodes += 1

            sentence_feature = torch.zeros(self.extra_feature_size, dtype=torch.float)
            sentence_feature[0] = float(sentence_len)
            sentence_feature[1] = float(num_nodes)
            sentence_feature[3] = 1.0
            sentence_feature[4] = 6.0
            scalar_features.append(sentence_feature.tolist())

            type_features.append([0.0] * self.num_edit_types)
            hyp_features.append([0.0] * self.num_hyps)
            labels.append(-999)
            span_features.append([-1, -1])
            if self.context_feature_size > 0 and context_features is not None:
                context_features.append([0.0] * self.context_feature_size)

            for node in range(num_nodes):
                src.extend([node, global_idx])
                dst.extend([global_idx, node])
                rel_types.extend([self.relation_types['global'], self.relation_types['global']])

        for node in range(total_nodes):
            src.append(node)
            dst.append(node)
            rel_types.append(self.relation_types['self_loop'])
        g = dgl.graph((src, dst), num_nodes=total_nodes)
        rel_tensor = torch.tensor(rel_types, dtype=torch.int64)

        # MODIFIED: Removed all flat_indices and offsets logic.
        # Now we assign the per-node multi-hot feature tensors.
        g.ndata['scalar_feat'] = torch.tensor(scalar_features, dtype=torch.float)
        g.ndata['type_feat'] = torch.tensor(type_features, dtype=torch.float)
        g.ndata['hyp_feat'] = torch.tensor(hyp_features, dtype=torch.float)
        if self.context_feature_size > 0 and context_features is not None:
            g.ndata['context_feat'] = torch.tensor(context_features, dtype=torch.float)
        g.ndata['span'] = torch.tensor(span_features, dtype=torch.long)
        
        mask_values = [False if (l is None or l == -999) else True for l in labels]
        if self.use_global_node and len(mask_values) > 0:
            mask_values[-1] = False
        label_values = [0.0 if (l is None or l == -999) else float(l) for l in labels]
        g.ndata['label'] = torch.tensor(label_values, dtype=torch.float)
        g.ndata['mask'] = torch.tensor(mask_values, dtype=torch.bool)
        g.edata['rel_type'] = rel_tensor
        return g

    def _build_edges(self, edit_keys, hyp_sets):
        src = []
        dst = []
        rel_types = []
        num_nodes = len(edit_keys)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                rel = self._relation_type(edit_keys[i], edit_keys[j], hyp_sets[i], hyp_sets[j])
                if rel is None:
                    continue
                src.extend([i, j])
                dst.extend([j, i])
                rel_types.extend([rel, rel])
        return src, dst, rel_types

    def _relation_type(self, edit_i, edit_j, hyps_i, hyps_j):
        if hyps_i.intersection(hyps_j):
            return self.relation_types['same_hypothesis']
        s1, e1, rep1 = edit_i
        s2, e2, rep2 = edit_j
        same_span = s1 == s2 and e1 == e2
        if same_span and rep1 != rep2:
            return self.relation_types['same_span']
        insertion_conflict = s1 == e1 == s2 == e2
        if insertion_conflict:
            return self.relation_types['insertion']
        
        is_adjacent = (e1 == s2 and not (s1 == e1 or s2 == e2)) or \
                      (e2 == s1 and not (s2 == e2 or s1 == e1))
        if is_adjacent:
            return self.relation_types['adjacent']

        intersecting_range = ((s1 <= s2 < e1) and not (s1 == e2)) or \
                             ((s2 <= s1 < e2) and not (s2 == e1))
        if intersecting_range:
            return self.relation_types['overlap']
        return None

    def _get_sentence_context(self, tokens: List[str]):
        if self.contextualizer is None:
            return None
        cache_key = tuple(tokens)
        if cache_key not in self.context_cache:
            self.context_cache[cache_key] = self.contextualizer.encode(tokens)
        return self.context_cache[cache_key]

    def _context_vector(self, sentence_context: Optional[torch.Tensor], span_start: int, span_end: int):
        if sentence_context is None or sentence_context.shape[0] == 0:
            return torch.zeros(self.context_feature_size, dtype=torch.float)
        length = sentence_context.shape[0]
        device = sentence_context.device
        if span_start < span_end:
            start = max(0, min(span_start, length - 1))
            end = max(start + 1, min(span_end, length))
            indices = torch.arange(start, end, dtype=torch.long, device=device)
        else:
            left = max(0, min(span_start - 1, length - 1))
            right = max(0, min(span_start, length - 1))
            indices = torch.tensor([left, right], dtype=torch.long, device=device)
        context_vec = sentence_context.index_select(0, indices)
        return context_vec.mean(dim=0).cpu()

    def label_counts(self):
        label_count = [0, 0]
        for graph in self.graphs:
            mask = graph.ndata['mask']
            labels = graph.ndata['label'][mask]
            if labels.numel() == 0:
                continue
            positives = labels.sum().item()
            total = labels.numel()
            label_count[1] += positives
            label_count[0] += total - positives
        self.label_count = label_count

    # MODIFIED: transform now creates multi-hot vectors
    def transform(self, data, edit_types, test=False):
        data = zip(*data)
        for entity in data:
            hyps = list(entity)
            assert min([hyps[0]['source'] == h['source'] for h in hyps]), "Sources are different!"
            sentence_tokens = hyps[0]['source'].split()
            sentence_len = max(1, len(sentence_tokens))

            en_edits = OrderedDict()
            for h_idx, hyp in enumerate(hyps):
                h_edits = hyp['edits']
                if 'labels' in hyp:
                    h_labels = hyp['labels']
                else:
                    h_labels = [None] * len(h_edits)
                for edit, label in zip(h_edits, h_labels):
                    e_start, e_end, e_type, e_cor = edit
                    edit_key = (e_start, e_end, e_cor)
                    if edit_key not in en_edits:
                        en_edits[edit_key] = [(h_idx, e_type, label)]
                    else:
                        en_edits[edit_key].append((h_idx, e_type, label))

            # MODIFIED: Create lists for multi-hot features
            en_scalar_features = []
            en_type_features = []
            en_hyp_features = []
            en_labels = []
            hyp_sets = []            # Still needed for _build_edges
            sentence_context = self._get_sentence_context(sentence_tokens)
            en_context_features = [] if self.context_feature_size > 0 else None

            for edit_key, edits in en_edits.items():
                scalar_feature = [0.0] * self.extra_feature_size
                e_label = -999
                hyp_indices = set()
                current_edit_type_indices = set() 

                for edit in edits:
                    h_idx, e_type, label = edit
                    hyp_indices.add(h_idx)
                    if e_type in edit_types:
                        current_edit_type_indices.add(edit_types[e_type])
                        
                    if label is not None:
                        if e_label == -999:
                            e_label = label
                        else:
                            assert e_label == label, "Labels are different"
                
                # NEW: Create multi-hot vectors
                type_feature_vec = [0.0] * self.num_edit_types
                for idx in current_edit_type_indices:
                    type_feature_vec[idx] = 1.0
                
                hyp_feature_vec = [0.0] * self.num_hyps
                for idx in hyp_indices:
                    hyp_feature_vec[idx] = 1.0

                # ... (Scalar feature logic is unchanged) ...
                span_start, span_end, replacement = edit_key
                span_length = max(1, span_end - span_start)
                span_norm = float(span_length) / float(sentence_len)
                is_insertion = 1.0 if span_start == span_end else 0.0
                is_deletion = 1.0 if len(replacement.strip()) == 0 and span_start < span_end else 0.0
                hyp_coverage = float(len(hyp_indices)) / max(1, len(hyps))
                if hyp_coverage <= 0.0:
                    teacher_logit = -6.0
                elif hyp_coverage >= 1.0:
                    teacher_logit = 6.0
                else:
                    teacher_logit = math.log(hyp_coverage / (1.0 - hyp_coverage))
                
                scalar_feature[0] = span_norm
                scalar_feature[1] = is_insertion
                scalar_feature[2] = is_deletion
                scalar_feature[3] = hyp_coverage
                scalar_feature[4] = teacher_logit

                # MODIFIED: Append all feature types
                en_scalar_features.append(scalar_feature)
                en_type_features.append(type_feature_vec)
                en_hyp_features.append(hyp_feature_vec)
                en_labels.append(e_label)
                hyp_sets.append(hyp_indices)

                if en_context_features is not None:
                    context_vec = self._context_vector(
                        sentence_context,
                        span_start,
                        span_end
                    )
                    en_context_features.append(context_vec.tolist())
            
            # MODIFIED: Call _build_graph with the new signature
            graph = self._build_graph(en_scalar_features, en_type_features, en_hyp_features,
                                      en_context_features,
                                      en_labels, list(en_edits.keys()),
                                      hyp_sets, sentence_len)
            
            if self.test:
                self.all_edits.append(
                    {'source': hyps[0]['source'], 'edits': en_edits}
                )
            if len(en_scalar_features) == 0 and not self.test:
                continue
            self.graphs.append(graph)
            self.graph_edit_keys.append(list(en_edits.keys()))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def _conflicts(edit_a, edit_b):
    multiple_insertion = (edit_a[0] == edit_a[1] == edit_b[0] == edit_b[1])
    intersecting_range = ((edit_a[0] <= edit_b[0] < edit_a[1] and not edit_a[0] == edit_b[1]) or
                          (edit_b[0] <= edit_a[0] < edit_b[1] and not edit_b[0] == edit_a[1]))
    return multiple_insertion or intersecting_range


def _filter_conflicts(candidates):
    filtered = []
    for item in candidates:
        if all(not _conflicts(item[0], sel[0]) for sel in filtered):
            filtered.append(item)
    return filtered


def select_edits_beam(edit_items, keep_probs, priority_scores, beam_size,
                      min_prob, priority_weight):
    if beam_size <= 1:
        return []
    candidates = []
    for edit, prob, priority in zip(edit_items, keep_probs, priority_scores):
        p_val = float(prob)
        if p_val < min_prob:
            continue
        candidates.append((edit, p_val, float(priority)))
    if not candidates:
        return []
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    beams = [([], 0.0)]
    for edit, prob, priority in candidates:
        new_beams = []
        for selected, score in beams:
            new_beams.append((selected, score))
            if any(_conflicts(edit, sel[0]) for sel in selected):
                continue
            logit = math.log(prob + 1e-6) - math.log(max(1e-6, 1.0 - prob + 1e-6))
            new_score = score + logit + priority_weight * priority
            new_beams.append((selected + [((edit), prob, priority)], new_score))
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]
    best = max(beams, key=lambda x: x[1])[0]
    return sorted(best, key=lambda x: (x[2], x[1]), reverse=True)


def select_edits_threshold(edit_items, keep_probs, priority_scores, threshold):
    candidates = []
    for edit, prob, priority in zip(edit_items, keep_probs, priority_scores):
        p_val = float(prob)
        if p_val >= threshold:
            candidates.append((edit, p_val, float(priority)))
    candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return _filter_conflicts(candidates)


def _pairwise_priority_loss_single(logits, spans, labels, margin):
    span_list = spans.tolist()
    label_list = labels.tolist()
    pos_idx = [i for i, l in enumerate(label_list) if l > 0.5]
    neg_idx = [i for i, l in enumerate(label_list) if l <= 0.5]
    pair_losses = []
    for pi in pos_idx:
        for ni in neg_idx:
            if _conflicts((span_list[pi][0], span_list[pi][1], None),
                          (span_list[ni][0], span_list[ni][1], None)):
                diff = logits[pi] - logits[ni]
                pair_losses.append(F.softplus(-(diff - margin)))
    if not pair_losses:
        return None
    return torch.stack(pair_losses).mean()


def priority_pairwise_loss(graph, priority_logits, margin):
    spans = graph.ndata['span']
    labels = graph.ndata['label']
    mask = graph.ndata['mask']
    batch_nodes = graph.batch_num_nodes()
    offset = 0
    losses = []
    for num in batch_nodes.tolist():
        if num == 0:
            continue
        idx = torch.arange(offset, offset + num, device=priority_logits.device)
        valid_idx = idx[mask[idx]]
        if valid_idx.numel() == 0:
            offset += num
            continue
        g_logits = priority_logits[valid_idx]
        g_spans = spans[valid_idx]
        g_labels = labels[valid_idx]
        single_loss = _pairwise_priority_loss_single(g_logits, g_spans, g_labels, margin)
        if single_loss is not None:
            losses.append(single_loss)
        offset += num
    if not losses:
        return torch.tensor(0.0, device=priority_logits.device)
    return torch.stack(losses).mean()


class RelGraphBlock(nn.Module):
    """
    Relational GNN block with residual connections and a position-wise feed-forward layer.
    """
    def __init__(self, dim, num_relations, dropout, ff_multiplier):
        super().__init__()
        self.conv = RelGraphConv(dim, dim, num_relations)
        self.pre_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)
        hidden_ff = max(dim, int(dim * ff_multiplier))
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_ff, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, x, edge_type):
        h = self.conv(graph, x, edge_type)
        h = F.gelu(self.pre_norm(h))
        x = x + self.dropout(h)
        ff_output = self.ff(self.ff_norm(x))
        x = x + self.dropout(ff_output)
        return x


class ContextBlock(nn.Module):
    """
    Lightweight transformer-style block applied inside each graph to model global competition
    between edits beyond the explicit RGCN edges.
    """
    def __init__(self, hidden_dim, num_heads, dropout, ff_multiplier):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        hidden_ff = max(hidden_dim, int(hidden_dim * ff_multiplier))
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_ff, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        if x.shape[0] == 0:
            return x
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        return x


class GNNModel(nn.Module):
    """
    Relation-aware GNN enhanced with feature encoders, residual R-GCN blocks,
    and per-graph self-attention refinement for stronger global reasoning.
    """
    def __init__(self, num_edit_types, num_hyps, num_relations,
                 scalar_feature_size=16, edit_embed_dim=32, hyp_embed_dim=16,
                 scalar_embed_dim=16, context_feature_size=0, context_proj_dim=128,
                 hidden_dim=128, num_layers=2,
                 dropout=0.1, ff_hidden_multiplier=2.0,
                 context_layers=1, context_heads=4,
                 context_mask_rel_ids=None, context_mask_bias=-1e4):
        super().__init__()
        assert num_layers >= 1, "Number of GNN layers must be >= 1"
        assert context_heads >= 1, "context_heads must be >= 1"

        self.has_context = context_feature_size > 0
        self.context_mask_rel_ids = set(context_mask_rel_ids or [])
        self.context_mask_bias = context_mask_bias

        self.scalar_encoder = nn.Sequential(
            nn.LayerNorm(scalar_feature_size),
            nn.Linear(scalar_feature_size, scalar_embed_dim),
            nn.GELU(),
        )
        self.edit_type_encoder = nn.Sequential(
            nn.Linear(num_edit_types, edit_embed_dim, bias=False),
            nn.GELU(),
        )
        self.hyp_encoder = nn.Sequential(
            nn.Linear(num_hyps, hyp_embed_dim, bias=False),
            nn.GELU(),
        )

        feature_length = scalar_embed_dim + edit_embed_dim + hyp_embed_dim
        if self.has_context:
            self.context_encoder = nn.Sequential(
                nn.LayerNorm(context_feature_size),
                nn.Linear(context_feature_size, context_proj_dim),
                nn.GELU(),
            )
            feature_length += context_proj_dim
        else:
            self.context_encoder = None

        self.feature_fusion = nn.Sequential(
            nn.LayerNorm(feature_length),
            nn.Linear(feature_length, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.graph_blocks = nn.ModuleList([
            RelGraphBlock(hidden_dim, num_relations, dropout, ff_hidden_multiplier)
            for _ in range(num_layers)
        ])
        self.context_blocks = nn.ModuleList([
            ContextBlock(hidden_dim, context_heads, dropout, ff_hidden_multiplier)
            for _ in range(context_layers)
        ])

        self.keep_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.priority_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _get_batch_segments(self, graph, total_nodes):
        try:
            counts = graph.batch_num_nodes()
        except AttributeError:
            counts = None
        if counts is None:
            return [(0, total_nodes)]
        if torch.is_tensor(counts):
            counts = counts.tolist()
        segments = []
        start = 0
        for count in counts:
            end = start + int(count)
            segments.append((start, end))
            start = end
        if not segments:
            segments = [(0, total_nodes)]
        return segments

    def _build_attention_mask(self, start, end, src, dst, rel_type, device):
        if not self.context_mask_rel_ids:
            return None
        length = end - start
        if length <= 1:
            return None
        src_mask = (src >= start) & (src < end) & (dst >= start) & (dst < end)
        if not torch.any(src_mask):
            return None
        local_src = (src[src_mask] - start).tolist()
        local_dst = (dst[src_mask] - start).tolist()
        local_rel = rel_type[src_mask].tolist()
        mask = torch.zeros((length, length), dtype=torch.float, device=device)
        has_mask = False
        for s_local, d_local, rel in zip(local_src, local_dst, local_rel):
            if int(rel) in self.context_mask_rel_ids:
                mask[d_local, s_local] = self.context_mask_bias
                mask[s_local, d_local] = self.context_mask_bias
                has_mask = True
        return mask if has_mask else None

    def _apply_context_blocks(self, x, graph):
        if not self.context_blocks or x.shape[0] == 0:
            return x
        segments = self._get_batch_segments(graph, x.shape[0])
        src, dst = graph.edges(order='eid')
        rel_type = graph.edata['rel_type']
        for block in self.context_blocks:
            updated = []
            for start, end in segments:
                chunk = x[start:end]
                if chunk.shape[0] == 0:
                    updated.append(chunk)
                    continue
                attn_mask = self._build_attention_mask(start, end, src, dst, rel_type, chunk.device)
                chunk = chunk.unsqueeze(1)  # (seq_len, batch=1, hidden)
                chunk = block(chunk, attn_mask=attn_mask)
                updated.append(chunk.squeeze(1))
            x = torch.cat(updated, dim=0)
        return x

    def forward(self, graph, features=None):
        if graph.num_nodes() == 0:
            empty = graph.ndata['scalar_feat'].new_zeros(0)
            return empty, empty

        if features is not None:
            x = features
        else:
            scalar_feat = graph.ndata['scalar_feat']
            type_feat = graph.ndata['type_feat']
            hyp_feat = graph.ndata['hyp_feat']

            scalar_emb = self.scalar_encoder(scalar_feat)
            type_emb = self.edit_type_encoder(type_feat)
            hyp_emb = self.hyp_encoder(hyp_feat)
            fused_inputs = [scalar_emb, type_emb, hyp_emb]
            if self.has_context and 'context_feat' in graph.ndata:
                context_emb = self.context_encoder(graph.ndata['context_feat'])
                fused_inputs.append(context_emb)
            fused = torch.cat(fused_inputs, dim=1)
            x = self.feature_fusion(fused)

        edge_type = graph.edata['rel_type']
        for block in self.graph_blocks:
            x = block(graph, x, edge_type)
        x = self._apply_context_blocks(x, graph)

        keep_logits = self.keep_head(x).squeeze(-1)
        priority_logits = self.priority_head(x).squeeze(-1)
        return keep_logits, priority_logits


def train(model, train_dataset, batch_size, lr, weight_decay, num_epoch, device,
            model_path=None, eval_dataset=None, save_last=False, verbose=True,
            priority_loss_weight=0.5, priority_pair_margin=0.0, early_stop_patience=0):
    """
    Train the model and save the best checkpoint
    """
    data_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    start = datetime.datetime.now()
    metric = 'f0.5'
    best_score = 0
    best_epoch = 0
    epochs_since_improve = 0
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        steps_with_loss = 0
        for step, graph in enumerate(data_loader, 0):
            graph = graph.to(device)
            labels = graph.ndata['label']
            mask = graph.ndata['mask']
            if mask.sum() == 0:
                continue

            optimizer.zero_grad()

            keep_logits, priority_logits = model(graph)
            loss_main = criterion(keep_logits[mask], labels[mask])
            if priority_loss_weight > 0:
                loss_aux = priority_pairwise_loss(graph, priority_logits, priority_pair_margin)
                loss = loss_main + priority_loss_weight * loss_aux
            else:
                loss = loss_main

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            running_loss += loss.item()
            steps_with_loss += 1

            if verbose and step % 100 == 99 and steps_with_loss > 0:
                print('[%d, %3d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / steps_with_loss))
                running_loss = 0.0
                steps_with_loss = 0
        if eval_dataset is not None:
            result = eval(model, eval_dataset, device)
            if metric in result:
                score = result[metric]
                if verbose:
                    print('[{}] Accuracy: {}, F0.5: {}'.format(epoch, result['acc'], result['f0.5']))
                if score > best_score:
                    best_score = score
                    best_epoch = epoch # 0-based index
                    epochs_since_improve = 0
                    if model_path is not None:
                        checkpoint = {
                            'edit_types': train_dataset.edit_types,
                            'hyp_list': train_dataset.hyp_list,
                            'model_state_dict': model.state_dict()
                        }
                        torch.save(checkpoint, model_path)
                        print('Model with {} accuracy saved on {}'.format(score, model_path))
                else:
                    if early_stop_patience > 0:
                        epochs_since_improve += 1
                        if epochs_since_improve >= early_stop_patience:
                            if verbose:
                                print('Early stopping triggered after {} epochs without improvement.'.format(epochs_since_improve))
                            break
            else:
                if verbose:
                    print('No accuracy found. No model will be saved.')
    end = datetime.datetime.now()
    if save_last:
        checkpoint = {
            'edit_types': train_dataset.edit_types,
            'hyp_list': train_dataset.hyp_list,
            'model_state_dict': model.state_dict()
        }
        torch.save(checkpoint, model_path)
    if verbose:
        print('== best checkpoint ({}) from epoch {} saved in {}'.format(best_score, best_epoch, model_path))
        print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))

    return best_score, best_epoch


def finetune_with_cosine(model, dataset, batch_size, lr, weight_decay, num_epoch,
                         device, priority_loss_weight=0.5, priority_pair_margin=0.0,
                         eta_min=0.0, model_path=None, verbose=True, dataset_meta=None):
    if num_epoch <= 0:
        return
    data_loader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epoch), eta_min=eta_min
    )
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        steps = 0
        for graph in data_loader:
            graph = graph.to(device)
            labels = graph.ndata['label']
            mask = graph.ndata['mask']
            if mask.sum() == 0:
                continue
            optimizer.zero_grad()
            keep_logits, priority_logits = model(graph)
            loss_main = criterion(keep_logits[mask], labels[mask])
            if priority_loss_weight > 0:
                loss_aux = priority_pairwise_loss(graph, priority_logits, priority_pair_margin)
                loss = loss_main + priority_loss_weight * loss_aux
            else:
                loss = loss_main
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            steps += 1
        scheduler.step()
        if verbose and steps > 0:
            print(f"[cosine finetune] epoch {epoch+1}/{num_epoch} loss: {running_loss / steps:.4f} lr: {scheduler.get_last_lr()[0]:.6f}")
    if model_path is not None:
        checkpoint = {'model_state_dict': model.state_dict()}
        if dataset_meta:
            checkpoint.update(dataset_meta)
        torch.save(checkpoint, model_path)


def eval(model, dataset, device='cpu'):
    """
    Evaluation function to get an estimated F0.5 score to save
    the best checkpoint during training.
    """
    model.eval()
    data_loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        tp = 0
        tn = 0
        p = 0
        true_edits = 0
        total_data = 0
        result = {
            'preds': []
        }
        for graph in data_loader:
            graph = graph.to(device)
            labels = graph.ndata['label']
            mask = graph.ndata['mask']
            keep_logits, priority_logits = model(graph)
            outputs = torch.sigmoid(keep_logits)
            preds = torch.round(outputs)
            result['preds'].append(preds.cpu())
            if mask.sum() == 0:
                continue
            preds_masked = preds[mask]
            labels_masked = labels[mask]
            p += torch.sum(preds_masked).item()
            true_edits += torch.sum(labels_masked).item()
            tp += torch.sum((preds_masked > 0) & (labels_masked > 0)).item()
            tn += torch.sum((preds_masked == 0) & (labels_masked == 0)).item()
            total_data += labels_masked.numel()
                
        precision = 1 if p == 0 else float(tp) / p
        recall = 1 if true_edits == 0 else float(tp) / true_edits
        f_half = 0 if precision + recall == 0 else (1 + 0.5 * 0.5) * precision * recall / (0.5 * 0.5 * precision + recall)
        if result['preds']:
            result['preds'] = torch.cat(result['preds'])
        else:
            result['preds'] = torch.tensor([])
        acc = float(tp + tn) / total_data if total_data > 0 else 1.0
        result['acc'] = acc
        result['prec'] = precision
        result['rec'] = recall
        result['f0.5'] = f_half

    return result

def test(model, model_path, dataset, device, threshold=0.5, generate_text=True,
         beam_size=1, beam_priority_weight=0.0, beam_min_prob=0.0):
    """
    A test function to predict the appropriate edit and apply it
    to the original sentence, resulting a corrected sentence
    """
    model_paths = model_path.split(',')
    raw_data = dataset.all_edits
    graphs = [dataset[i] for i in range(len(dataset))]
    
    result = [None] * len(graphs)
    ensemble_inputs = []
    with torch.no_grad():
        for path in model_paths:
            print('Getting predictions from {}...'.format(path))
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model_output = []
            for idx, graph in enumerate(graphs):
                edits = raw_data[idx]['edits']
                if graph.num_nodes() == 0 or len(edits) == 0:
                    model_output.append(None)
                    continue
                keep_logits, priority_logits = model(graph.to(device))
                keep_probs = torch.sigmoid(keep_logits).cpu()
                priority_scores = priority_logits.cpu()
                keep_probs = keep_probs[:len(edits)]
                priority_scores = priority_scores[:len(edits)]
                model_output.append((keep_probs, priority_scores))
            ensemble_inputs.append(model_output)
    
    all_outputs = []
    for outputs in zip(*ensemble_inputs):
        if outputs[0] is None:
            all_outputs.append(None)
            continue
        keep_stack = torch.stack([o[0] for o in outputs], dim=0)
        priority_stack = torch.stack([o[1] for o in outputs], dim=0)
        all_outputs.append((keep_stack.mean(dim=0), priority_stack.mean(dim=0)))

    if generate_text:
        for idx, output in enumerate(all_outputs):
            if output is None:
                result[idx] = raw_data[idx]['source']
                continue
            source = raw_data[idx]['source'].split()
            edits = raw_data[idx]['edits']
            offset = 0
            keep_probs, priority_scores = output
            edit_list = list(edits.keys())
            keep_list = keep_probs.detach().cpu().tolist()
            priority_list = priority_scores.detach().cpu().tolist()
            min_prob = max(threshold, beam_min_prob)
            if beam_size > 1:
                selected = select_edits_beam(edit_list, keep_list, priority_list,
                                             beam_size, min_prob, beam_priority_weight)
            else:
                selected = select_edits_threshold(edit_list, keep_list, priority_list, threshold)
            filtered_edits = sorted([
                (e[0][0], e[0][1], e[0][2], e[1], e[2]) for e in selected
            ])

            for edit in filtered_edits:
                e_start, e_end, rep_token, prob, priority = edit
                e_cor = rep_token.split()
                len_cor = 0 if len(rep_token) == 0 else len(e_cor)
                source[e_start + offset:e_end + offset] = e_cor
                offset = offset - (e_end - e_start) + len_cor
            result[idx] = ' '.join(source)
    else:
        for idx, output in enumerate(all_outputs):
            if output is None:
                result[idx] = []
                continue
            edits = raw_data[idx]['edits']
            keep_probs, priority_scores = output
            edits_to_apply = []
            for edit, prob, priority in zip(edits.keys(), keep_probs, priority_scores):
                e_start, e_end, rep_token = edit
                edits_to_apply.append((e_start, e_end, rep_token, prob.item(), priority.item()))
            result[idx] = edits_to_apply

    return result


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    allowed_hypotheses = None
    if args.hypotheses:
        allowed_hypotheses = [h.strip() for h in args.hypotheses.split(',') if h.strip()]

    device_str = 'cpu'
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)

    device = torch.device(device_str)

    mask_relation_names = args.context_mask_relations or []
    if len(mask_relation_names) == 1 and mask_relation_names[0].lower() == 'none':
        mask_relation_names = []

    contextualizer = None
    if args.contextual_encoder:
        if args.contextual_encoder_device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError("CUDA contextual encoder requested but CUDA is not available.")
        contextualizer = FrozenContextualEncoder(
            args.contextual_encoder,
            layer=args.contextual_encoder_layer,
            max_length=args.contextual_encoder_max_len,
            device=args.contextual_encoder_device,
        )

    use_global_node = not args.no_global_node

    if args.train:
        edit_types = create_vocab(args.m2_dir,
                                    args.data_dir,
                                    args.source_name,
                                    args.target_name
                                )
        hyp_list = [f for f in listdir(args.data_dir) if isfile(join(args.data_dir, f)) \
            and basename(f) not in [args.source_name, args.target_name]]
        vocab = {
            'edit_types': edit_types,
            'hyp_list': hyp_list,
        }
        with open(args.vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, indent=2)
        kf = KFold(n_splits=args.val_ratio, shuffle=True, random_state=args.seed)
        dummy_file = [1 for _ in open(join(args.data_dir, args.source_name), encoding='utf-8')]

        _BATCH_SIZE = 128
        _LR = args.lr
        _EPOCH = 200

        split = kf.split(dummy_file)
        train_index, test_index = next(split)
        
        # get number of epoch
        train_dataset = M2Dataset(args.m2_dir,
                                args.data_dir,
                                args.source_name,
                                args.target_name,
                                vocab,
                                filter_idx=train_index,
                                upsample=args.upsample,
                                max_hypotheses=args.max_hypotheses,
                                allowed_hypotheses=allowed_hypotheses,
                                contextualizer=contextualizer,
                                use_global_node=use_global_node,
                                )
        train_mask_rel_ids = [train_dataset.relation_types[r] for r in mask_relation_names
                              if r in train_dataset.relation_types]
        model = GNNModel(
                         num_edit_types=len(vocab['edit_types']),
                         num_hyps=train_dataset.num_hyps,
                         num_relations=train_dataset.num_relations,
                         scalar_feature_size=train_dataset.extra_feature_size,
                         edit_embed_dim=args.edit_type_dim,
                         hyp_embed_dim=args.hypothesis_dim,
                         scalar_embed_dim=args.scalar_feat_dim,
                         context_feature_size=train_dataset.context_feature_size,
                         context_proj_dim=args.contextual_proj_dim,
                         hidden_dim=args.gnn_hidden_dim,
                         num_layers=args.gnn_layers,
                         dropout=args.gnn_dropout,
                         ff_hidden_multiplier=args.ff_multiplier,
                         context_layers=args.context_layers,
                         context_heads=args.context_heads,
                         context_mask_rel_ids=train_mask_rel_ids,
                         context_mask_bias=args.context_mask_bias).to(device)
        eval_dataset = M2Dataset(args.m2_dir,
                                args.data_dir,
                                args.source_name,
                                args.target_name,
                                vocab,
                                filter_idx=test_index,
                                max_hypotheses=args.max_hypotheses,
                                allowed_hypotheses=allowed_hypotheses,
                                contextualizer=contextualizer,
                                use_global_node=use_global_node,
                                )

        _score, best_epoch = train(model, train_dataset, _BATCH_SIZE, _LR, args.weight_decay, _EPOCH,
                device, eval_dataset=eval_dataset, priority_loss_weight=args.priority_loss_weight,
                priority_pair_margin=args.priority_pair_margin,
                early_stop_patience=args.early_stop_patience)
        # full training
        torch.manual_seed(args.seed)
        print('Best checkpoint at epoch {}. Training on full dataset.'.format(best_epoch))
        model_path = join(args.model_path, 'model.pt')
        train_dataset = M2Dataset(args.m2_dir,
                                args.data_dir,
                                args.source_name,
                                args.target_name,
                                vocab,
                                upsample=args.upsample,
                                max_hypotheses=args.max_hypotheses,
                                allowed_hypotheses=allowed_hypotheses,
                                contextualizer=contextualizer,
                                use_global_node=use_global_node,
                                )
        full_mask_rel_ids = [train_dataset.relation_types[r] for r in mask_relation_names
                             if r in train_dataset.relation_types]
        model = GNNModel(
                        num_edit_types=len(vocab['edit_types']),
                        num_hyps=train_dataset.num_hyps,
                        num_relations=train_dataset.num_relations,
                        scalar_feature_size=train_dataset.extra_feature_size,
                        edit_embed_dim=args.edit_type_dim,
                        hyp_embed_dim=args.hypothesis_dim,
                        scalar_embed_dim=args.scalar_feat_dim,
                        context_feature_size=train_dataset.context_feature_size,
                        context_proj_dim=args.contextual_proj_dim,
                        hidden_dim=args.gnn_hidden_dim,
                        num_layers=args.gnn_layers,
                        dropout=args.gnn_dropout,
                        ff_hidden_multiplier=args.ff_multiplier,
                        context_layers=args.context_layers,
                        context_heads=args.context_heads,
                        context_mask_rel_ids=full_mask_rel_ids,
                        context_mask_bias=args.context_mask_bias).to(device)
        train(model, train_dataset, _BATCH_SIZE, _LR, args.weight_decay, best_epoch,
                device, model_path, save_last=True, priority_loss_weight=args.priority_loss_weight,
                priority_pair_margin=args.priority_pair_margin)
        if args.cosine_tail_epochs > 0:
            print(f"Running cosine LR finetune for {args.cosine_tail_epochs} epochs...")
            finetune_with_cosine(
                model, train_dataset, _BATCH_SIZE, _LR, args.weight_decay,
                args.cosine_tail_epochs, device,
                priority_loss_weight=args.priority_loss_weight,
                priority_pair_margin=args.priority_pair_margin,
                eta_min=args.cosine_tail_min_lr,
                model_path=model_path,
                dataset_meta={
                    'edit_types': train_dataset.edit_types,
                    'hyp_list': train_dataset.hyp_list,
                }
            )
        print('Finished training.')
    elif args.test or args.score:
        with open(args.vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        test_dataset = M2Dataset(args.m2_dir,
                                 args.data_dir,
                                 args.source_name,
                                 args.target_name,
                                 vocab,
                                 test=True,
                                 max_hypotheses=args.max_hypotheses,
                                 allowed_hypotheses=allowed_hypotheses,
                                 contextualizer=contextualizer,
                                 use_global_node=use_global_node,
                                )
        test_mask_rel_ids = [test_dataset.relation_types[r] for r in mask_relation_names
                             if r in test_dataset.relation_types]
        model = GNNModel(
                         num_edit_types=len(vocab['edit_types']),
                         num_hyps=test_dataset.num_hyps,
                         num_relations=test_dataset.num_relations,
                         scalar_feature_size=test_dataset.extra_feature_size,
                         edit_embed_dim=args.edit_type_dim,
                         hyp_embed_dim=args.hypothesis_dim,
                         scalar_embed_dim=args.scalar_feat_dim,
                         context_feature_size=test_dataset.context_feature_size,
                         context_proj_dim=args.contextual_proj_dim,
                         hidden_dim=args.gnn_hidden_dim,
                         num_layers=args.gnn_layers,
                         dropout=args.gnn_dropout,
                         ff_hidden_multiplier=args.ff_multiplier,
                         context_layers=args.context_layers,
                         context_heads=args.context_heads,
                         context_mask_rel_ids=test_mask_rel_ids,
                         context_mask_bias=args.context_mask_bias).to(device)
        results = test(model, args.model_path, test_dataset, device, threshold=args.threshold,
                        generate_text=args.test, beam_size=args.beam_size,
                        beam_priority_weight=args.beam_priority_weight,
                        beam_min_prob=args.beam_min_prob)
        if args.score:
            results = [json.dumps(r) for r in results]
        
        with open(args.output_path, 'w', encoding='utf-8') as out:
            out.write('\n'.join(results))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='path to the data directory')
    parser.add_argument('--m2_dir', default='m2', help='path to the generated m2 files')
    parser.add_argument('--source_name', default='source.txt', help='The source filename')
    parser.add_argument('--target_name', default='target.txt', help='The target filename')
    parser.add_argument('--max_hypotheses', type=int, default=None,
                        help='Maximum number of hypothesis files to include')
    parser.add_argument('--hypotheses', type=str, default=None,
                        help='Comma-separated list of hypothesis filenames to use')
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0, help="weight decay (L2 penalty)")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--val_ratio', type=int, default=5, help="1/val_ratio of the data is for validation")
    parser.add_argument('--threshold', type=float, default=0.5, help="probability threshold")
    parser.add_argument('--beam_size', type=int, default=5,
                        help='beam size used during decoding (1 disables beam search)')
    parser.add_argument('--beam_priority_weight', type=float, default=0.0,
                        help='weight added to priority scores inside beam decoding')
    parser.add_argument('--beam_min_prob', type=float, default=0.2,
                        help='minimum probability for a candidate to enter the beam (before threshold)')
    parser.add_argument('--upsample', type=str, default=None, help='up-sample ratio of class 0:class 1')
    parser.add_argument('--gnn_hidden_dim', type=int, default=128, help='hidden dimension for the GNN layers')
    parser.add_argument('--gnn_layers', type=int, default=2, help='number of GraphSAGE layers')
    parser.add_argument('--gnn_dropout', type=float, default=0.1, help='dropout rate applied between GNN layers')
    parser.add_argument('--no_global_node', action='store_true',
                        help='disable the sentence-level global node when building graphs (for ablations)')
    parser.add_argument('--edit_type_dim', type=int, default=64, help='dimension used to encode edit-type multi-hot features')
    parser.add_argument('--hypothesis_dim', type=int, default=32, help='dimension used to encode hypothesis membership multi-hot features')
    parser.add_argument('--scalar_feat_dim', type=int, default=32, help='dimension used to encode scalar handcrafted features')
    parser.add_argument('--ff_multiplier', type=float, default=2.0, help='expansion factor in the feed-forward sublayers inside each GNN/context block')
    parser.add_argument('--context_layers', type=int, default=1, help='number of transformer-style context blocks applied per graph (0 disables)')
    parser.add_argument('--context_heads', type=int, default=4, help='number of attention heads inside each context block')
    parser.add_argument('--contextual_encoder', type=str, default="",
                        help='name or path of a Hugging Face encoder (e.g., roberta-base) for contextual token embeddings')
    parser.add_argument('--contextual_encoder_layer', type=int, default=-1,
                        help='layer index from the contextual encoder to extract (negative for reverse indexing)')
    parser.add_argument('--contextual_encoder_max_len', type=int, default=256,
                        help='maximum number of whitespace tokens processed per encoder chunk')
    parser.add_argument('--contextual_encoder_device', type=str, default='cuda',
                        help='device identifier for running the contextual encoder (cpu or cuda:N)')
    parser.add_argument('--contextual_proj_dim', type=int, default=128,
                        help='projection size applied to contextual embeddings before fusion')
    parser.add_argument('--context_mask_relations', nargs='*', default=[],
                        help='relation names that receive negative attention bias in context blocks (use "none" to disable)')
    parser.add_argument('--context_mask_bias', type=float, default=-1e4,
                        help='bias value added to attention logits for masked relations (negative suppresses attention)')
    parser.add_argument('--cosine_tail_epochs', type=int, default=5,
                        help='number of cosine-decay fine-tuning epochs run after the main training phase (0 disables)')
    parser.add_argument('--cosine_tail_min_lr', type=float, default=0.0,
                        help='final learning rate for cosine finetuning (eta_min)')
    parser.add_argument('--priority_loss_weight', type=float, default=0.5,
                        help='weight applied to the auxiliary priority head loss')
    parser.add_argument('--priority_pair_margin', type=float, default=0.0,
                        help='margin used for the pairwise priority ranking loss')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='number of eval epochs with no improvement before stopping (0 disables)')
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--test', default=False, action='store_true', help='test the model')
    parser.add_argument('--score', default=False, action='store_true', help='produce a score for each edit')
    parser.add_argument('--target_path', help='path to the target file during training')
    parser.add_argument('--vocab_path', default='vocab.idx', help='path to the vocab file')
    parser.add_argument('--model_path', required=True, help='path to the model directory')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)
