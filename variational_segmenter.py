import math
from collections import defaultdict
from typing import List, Dict, Tuple


class VariationalBayesianSegmenter:
    def __init__(
        self,
        alpha0: float = 1.0,
        alpha1: float = 1.0,
        max_word_length: int = 10,
        vi_iterations: int = 50,
        tolerance: float = 1e-4,
        min_iterations: int = 5,
        max_vocab_size: int = None,
    ) -> None:
        self.alpha0 = float(alpha0)
        self.alpha1 = float(alpha1)
        self.max_word_length = int(max_word_length)
        self.vi_iterations = int(vi_iterations)
        self.tolerance = float(tolerance)
        self.min_iterations = int(min_iterations)
        self.max_vocab_size = max_vocab_size

        # Learned / derived after fit
        self.word_to_index: Dict[str, int] = {}
        self.index_to_word: List[str] = []
        self.vocab_size: int = 0

        # Variational parameters
        self.phi_unigram: List[float] = []  # size V
        self.phi_bigram: List[List[float]] = []  # size (V) x (V), includes <s> and </s>

        # Cached expectations
        self.e_g0: List[float] = []  # E[G0] over words (size V)
        self.elog_transition: List[List[float]] = []  # E[log P(w|prev)]

        # Special tokens (added to vocab)
        self.start_token = "<s>"
        self.end_token = "</s>"

    # ----------------------------- Utilities -----------------------------

    @staticmethod
    def _digamma(x: float) -> float:
        # Simple, dependency-free digamma approximation
        # For x <= 0, reflect (not expected here since Dirichlet params > 0)
        if x <= 0.0:
            # Fallback; avoid domain error
            x = 1e-8
        result = 0.0
        # Increase x to > 6 using recurrence
        while x < 6.0:
            result -= 1.0 / x
            x += 1.0
        # Asymptotic expansion
        inv = 1.0 / x
        inv2 = inv * inv
        result += math.log(x) - 0.5 * inv - inv2 * (1.0 / 12.0) + inv2 * inv2 * (1.0 / 120.0)
        return result

    @staticmethod
    def _logsumexp(values: List[float]) -> float:
        if not values:
            return float("-inf")
        m = max(values)
        if m == float("-inf"):
            return m
        s = sum(math.exp(v - m) for v in values)
        return m + math.log(s)

    # --------------------------- Lattice builder ---------------------------

    def _build_vocab(self, utterances: List[str]) -> None:
        vocab = set()
        # Add special tokens
        vocab.add(self.start_token)
        vocab.add(self.end_token)
        for utt in utterances:
            n = len(utt)
            for i in range(n):
                # words of length at least 1
                for j in range(i + 1, min(n, i + self.max_word_length) + 1):
                    vocab.add(utt[i:j])
        # Optionally limit vocab size (keep shortest substrings first, then lexicographic)
        if self.max_vocab_size is not None and len(vocab) > self.max_vocab_size:
            substrings = [w for w in vocab if w not in {self.start_token, self.end_token}]
            substrings.sort(key=lambda x: (len(x), x))
            keep = set(substrings[: self.max_vocab_size - 2])
            vocab = keep | {self.start_token, self.end_token}
        self.index_to_word = sorted(vocab)
        self.word_to_index = {w: i for i, w in enumerate(self.index_to_word)}
        self.vocab_size = len(self.index_to_word)

    def _utterance_edges(self, utt: str) -> Tuple[List[Tuple[int, int, int]], List[List[int]], List[List[int]]]:
        # Returns:
        # - edges: list of (start, end, word_idx)
        # - edges_by_start[pos] -> list of edge indices starting at pos
        # - edges_by_end[pos] -> list of edge indices ending at pos
        n = len(utt)
        edges: List[Tuple[int, int, int]] = []
        edges_by_start: List[List[int]] = [[] for _ in range(n + 1)]
        edges_by_end: List[List[int]] = [[] for _ in range(n + 1)]
        for i in range(n):
            for j in range(i + 1, min(n, i + self.max_word_length) + 1):
                w = utt[i:j]
                if w in self.word_to_index:
                    widx = self.word_to_index[w]
                    eidx = len(edges)
                    edges.append((i, j, widx))
                    edges_by_start[i].append(eidx)
                    edges_by_end[j].append(eidx)
        return edges, edges_by_start, edges_by_end

    # --------------------------- Variational core --------------------------

    def _initialize_variational_params(self) -> None:
        V = self.vocab_size
        # Unigram Dirichlet params: alpha0 / V + 0
        base = self.alpha0 / float(V)
        self.phi_unigram = [base for _ in range(V)]
        # Bigram Dirichlet params for each previous word (including <s>)
        # phi_bigram[u][w] = alpha1 * E[G0(w)] + expected_count(u->w)
        # Initialize with base only. E[G0] initially uniform
        base_bigram = self.alpha1 / float(V)
        self.phi_bigram = [[base_bigram for _ in range(V)] for _ in range(V)]
        # Expectations cache
        self._refresh_expectations()

    def _refresh_expectations(self) -> None:
        # Update E[G0] and E[log P(w|u)] from current phi
        V = self.vocab_size
        start_idx = self.word_to_index.get(self.start_token, None)
        # E[G0] excluding start token as a possible next word
        sum_uni = float(sum(self.phi_unigram))
        if sum_uni <= 0:
            sum_uni = 1e-12
        raw = [p / sum_uni for p in self.phi_unigram]
        if start_idx is not None:
            raw_sum = sum(v for i, v in enumerate(raw) if i != start_idx)
            if raw_sum <= 0:
                raw_sum = 1e-12
            self.e_g0 = [(0.0 if i == start_idx else v / raw_sum) for i, v in enumerate(raw)]
        else:
            self.e_g0 = raw
        # E[log P(w|u)]
        self.elog_transition = [[0.0 for _ in range(V)] for _ in range(V)]
        for u in range(V):
            row = self.phi_bigram[u]
            denom = float(sum(row))
            if denom <= 0:
                denom = 1e-12
            digamma_denom = self._digamma(denom)
            for w in range(V):
                if start_idx is not None and w == start_idx:
                    # Disallow transitioning into start token
                    self.elog_transition[u][w] = -1e9
                else:
                    self.elog_transition[u][w] = self._digamma(row[w]) - digamma_denom

    def _forward_backward(
        self,
        utt: str,
        edges: List[Tuple[int, int, int]],
        edges_by_start: List[List[int]],
        edges_by_end: List[List[int]],
        start_idx: int,
        end_idx: int,
    ) -> Tuple[List[float], List[float], float, List[float], Dict[Tuple[int, int], float]]:
        # Returns:
        # - log_alpha per edge
        # - log_beta per edge
        # - logZ (normalizer)
        # - gamma per edge (marginal prob of taking edge)
        # - xi dict mapping (pred_edge_idx, edge_idx) to expected count
        n = len(utt)
        E = len(edges)
        log_alpha = [float("-inf")] * E
        log_beta = [float("-inf")] * E

        # Forward pass in order of increasing end position
        for end_pos in range(1, n + 1):
            for eidx in edges_by_end[end_pos]:
                s, t, w = edges[eidx]
                assert t == end_pos
                # predecessors: start if s == 0, else edges_by_end[s]
                contribs: List[float] = []
                if s == 0:
                    contribs.append(0.0 + self.elog_transition[start_idx][w])
                for pidx in edges_by_end[s]:
                    uprev = edges[pidx][2]
                    contribs.append(log_alpha[pidx] + self.elog_transition[uprev][w])
                log_alpha[eidx] = self._logsumexp(contribs) if contribs else float("-inf")

        # Backward pass in order of decreasing start position
        for start_pos in range(n, -1, -1):
            for eidx in edges_by_start[start_pos]:
                s, t, w = edges[eidx]
                # successors: if t == n, only end; else edges_by_start[t]
                contribs: List[float] = []
                if t == n:
                    contribs.append(self.elog_transition[w][end_idx])
                for nidx in edges_by_start[t]:
                    wnext = edges[nidx][2]
                    contribs.append(self.elog_transition[w][wnext] + log_beta[nidx])
                log_beta[eidx] = self._logsumexp(contribs) if contribs else float("-inf")

        # Normalizer Z
        end_contribs = []
        for eidx in edges_by_end[n]:
            w = edges[eidx][2]
            end_contribs.append(log_alpha[eidx] + self.elog_transition[w][end_idx])
        logZ = self._logsumexp(end_contribs)

        # Edge marginals gamma
        gamma = [0.0] * E
        if logZ == float("-inf"):
            # degenerate, no valid segmentation; return zeros
            xi = {}
            return log_alpha, log_beta, logZ, gamma, xi
        for eidx in range(E):
            gamma[eidx] = math.exp(log_alpha[eidx] + log_beta[eidx] - logZ)

        # Pairwise marginals xi for transitions pred->edge
        xi: Dict[Tuple[int, int], float] = {}
        # transitions from start -> first-word edges
        for eidx in edges_by_start[0]:
            w = edges[eidx][2]
            val = math.exp(0.0 + self.elog_transition[start_idx][w] + log_beta[eidx] - logZ)
            xi[(-1, eidx)] = val
        # transitions between edges
        for end_pos in range(1, n + 1):
            for eidx in edges_by_end[end_pos]:
                s, t, w = edges[eidx]
                for nidx in edges_by_start[t]:
                    wnext = edges[nidx][2]
                    val = math.exp(
                        log_alpha[eidx] + self.elog_transition[w][wnext] + log_beta[nidx] - logZ
                    )
                    xi[(eidx, nidx)] = val
        # transitions to end from last edges
        # we store these separately by using key (edge_idx, -1)
        for eidx in edges_by_end[n]:
            w = edges[eidx][2]
            val_end = math.exp(log_alpha[eidx] + self.elog_transition[w][end_idx] - logZ)
            xi[(eidx, -1)] = val_end

        return log_alpha, log_beta, logZ, gamma, xi

    # ------------------------------ Public API ----------------------------

    def fit(self, utterances: List[str]) -> None:
        if not utterances:
            raise ValueError("No utterances provided")

        # Build vocabulary and initialize variational params
        self._build_vocab(utterances)
        V = self.vocab_size
        start_idx = self.word_to_index[self.start_token]
        end_idx = self.word_to_index[self.end_token]

        self._initialize_variational_params()

        # Pre-build lattices for speed
        lattices = []
        for utt in utterances:
            edges, edges_by_start, edges_by_end = self._utterance_edges(utt)
            lattices.append((utt, edges, edges_by_start, edges_by_end))

        # VI loop
        prev_phi_unigram = self.phi_unigram[:]
        total_logZ = 0.0
        for it in range(self.vi_iterations):
            # E-step: expected counts under current elog_transition
            expected_unigram = [0.0 for _ in range(V)]
            expected_bigram = [[0.0 for _ in range(V)] for _ in range(V)]
            total_logZ = 0.0

            for (utt, edges, edges_by_start, edges_by_end) in lattices:
                if not edges:
                    # If no edges possible (e.g., empty utterance), skip
                    continue
                _, _, logZ, gamma, xi = self._forward_backward(
                    utt, edges, edges_by_start, edges_by_end, start_idx, end_idx
                )
                if logZ == float("-inf"):
                    # No valid segmentation path; skip contribution
                    continue
                total_logZ += logZ
                # Unigram expected counts over words
                for eidx, g in enumerate(gamma):
                    w = edges[eidx][2]
                    expected_unigram[w] += g
                # Bigram expected counts from xi
                for (pidx, eidx), val in xi.items():
                    if pidx == -1 and eidx >= 0:
                        # start -> first
                        w = edges[eidx][2]
                        expected_bigram[start_idx][w] += val
                    elif pidx >= 0 and eidx >= 0:
                        uprev = edges[pidx][2]
                        w = edges[eidx][2]
                        expected_bigram[uprev][w] += val
                    elif pidx >= 0 and eidx == -1:
                        # last -> end
                        uprev = edges[pidx][2]
                        expected_bigram[uprev][end_idx] += val

            # M-step (VB updates): update phi params
            # Unigram
            base_uni = self.alpha0 / float(V)
            for w in range(V):
                self.phi_unigram[w] = base_uni + expected_unigram[w]

            # Update base expectation E[G0]
            sum_uni = float(sum(self.phi_unigram))
            if sum_uni <= 0:
                sum_uni = 1e-12
            self.e_g0 = [p / sum_uni for p in self.phi_unigram]

            # Bigram
            for u in range(V):
                row = self.phi_bigram[u]
                # alpha1 * E[G0(w)] + expected counts
                for w in range(V):
                    row[w] = self.alpha1 * self.e_g0[w]
                # add expected counts
                eb_row = expected_bigram[u]
                for w in range(V):
                    row[w] += eb_row[w]

            # Refresh expectations
            self._refresh_expectations()

            # Convergence check based on unigram params
            if it + 1 >= self.min_iterations:
                max_rel_change = 0.0
                for a, b in zip(prev_phi_unigram, self.phi_unigram):
                    if a <= 0 and b <= 0:
                        continue
                    denom = max(1e-8, abs(a))
                    max_rel_change = max(max_rel_change, abs(b - a) / denom)
                if max_rel_change < self.tolerance:
                    break
            prev_phi_unigram = self.phi_unigram[:]

        # Cache for decoding
        self.utterances_ = utterances
        self.total_logZ_ = total_logZ

    def get_words(self, utterances: List[str] = None) -> List[List[str]]:
        """Segment all utterances (or those provided) using Viterbi under variational params."""
        if utterances is None:
            if not hasattr(self, "utterances_"):
                raise RuntimeError("Model has not been fit.")
            utterances = self.utterances_
        return [self.segment_utterance(utt) for utt in utterances]

    def segment_utterance(self, utt: str) -> List[str]:
        if not self.word_to_index:
            raise RuntimeError("Model has not been fit.")
        start_idx = self.word_to_index[self.start_token]
        end_idx = self.word_to_index[self.end_token]

        # Build edges
        edges, edges_by_start, edges_by_end = self._utterance_edges(utt)
        n = len(utt)
        E = len(edges)
        if E == 0:
            return []

        # Viterbi decoding over edges
        # best_score[eidx]: best log-prob ending at edge eidx
        # best_prev[eidx]: predecessor edge index (-1 means start)
        best_score = [float("-inf")] * E
        best_prev = [-2] * E

        # Initialization: edges starting at 0 from start
        for eidx in edges_by_start[0]:
            w = edges[eidx][2]
            best_score[eidx] = self.elog_transition[start_idx][w]
            best_prev[eidx] = -1

        # DP transitions between edges
        # Process by increasing end position
        for pos in range(1, n + 1):
            for eidx in edges_by_end[pos]:
                s, t, w = edges[eidx]
                # consider predecessors ending at s
                for pidx in edges_by_end[s]:
                    uprev = edges[pidx][2]
                    score = best_score[pidx] + self.elog_transition[uprev][w]
                    if score > best_score[eidx]:
                        best_score[eidx] = score
                        best_prev[eidx] = pidx

        # Termination: add transition to end token
        best_final_score = float("-inf")
        best_final_edge = -1
        for eidx in edges_by_end[n]:
            w = edges[eidx][2]
            score = best_score[eidx] + self.elog_transition[w][end_idx]
            if score > best_final_score:
                best_final_score = score
                best_final_edge = eidx

        if best_final_edge == -1:
            return []

        # Backtrace path of edges
        path_edges = []
        cur = best_final_edge
        while cur != -1 and cur != -2:
            path_edges.append(cur)
            cur = best_prev[cur]
        path_edges.reverse()

        # Convert edges to words
        words = []
        for eidx in path_edges:
            s, t, w = edges[eidx]
            words.append(utt[s:t])
        return words