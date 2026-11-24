"""Bounded local inference for a pinned Hugging Face tabular checkpoint."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

REPO = 'zerooneresearch/predictlm-base-26m'
REVISION = 'd4ededf7525e97ef9bc9ee1f3acac1ffbd3dbf97'


class PredictLMEstimator(BaseEstimator):
    def __init__(self, task='classification', fine_tune_steps=3):
        self.task = task
        self.fine_tune_steps = fine_tune_steps

    def fit(self, X, y):
        import torch
        from huggingface_hub import hf_hub_download
        from predictlm import PredictLM
        from predictlm.model import PredictLMv11, V11Config
        from sklearn.model_selection import train_test_split
        torch.set_num_threads(2)
        y = np.asarray(y)
        if self.task == 'classification' and len(np.unique(y)) > 10:
            raise ValueError('PredictLM supports at most 10 classes.')
        if len(y) > 256:
            indexes = np.arange(len(y))
            stratify = y if self.task == 'classification' else None
            indexes, _ = train_test_split(indexes, train_size=256, random_state=42, stratify=stratify)
            X, y = X[indexes], y[indexes]
        self.context_rows_ = len(y)
        self.selector_ = None
        if X.shape[1] > 32:
            self.selector_ = SelectKBest(f_classif if self.task == 'classification' else f_regression, k=32)
            X = self.selector_.fit_transform(X, y)
        self.context_features_ = X.shape[1]
        X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        path = hf_hub_download(REPO, 'v11_final.pt', revision=REVISION)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        cfg = V11Config(**payload['cfg'])
        net = PredictLMv11(cfg)
        incompatible = net.load_state_dict(payload.get('ema', payload['model']), strict=False)
        allowed_missing = {'y_embed.freqs', 'reg_head.bin_edges', 'reg_head.bin_centers'}
        if incompatible.unexpected_keys or set(incompatible.missing_keys) - allowed_missing:
            raise ValueError('Hugging Face weights do not match the model architecture.')
        self.model_ = PredictLM(net, cfg, device='cpu', auto_duo=False)
        self.model_.fit(X, y, task_type=self.task)
        # Optimize on disjoint context/query subsets drawn only from the training split.
        import torch.nn.functional as F
        from predictlm.heads import standardize_y_per_task, bar_distribution_loss
        rng = np.random.RandomState(42)
        context_x, context_y = self.model_._X_ctx, self.model_._y_ctx
        count = min(len(context_y), 80)
        n_context = max(1, int(count * .8))
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5)
        self.loss_curve_ = []
        net.train()
        for step in range(self.fine_tune_steps):
            indices = rng.permutation(len(context_y))[:count]
            ctx, query = indices[:n_context], indices[n_context:]
            x_ctx = torch.from_numpy(context_x[ctx]).float().unsqueeze(0)
            x_query = torch.from_numpy(context_x[query]).float().unsqueeze(0)
            mask = torch.zeros(1, context_x.shape[1], dtype=torch.bool)
            if self.task == 'classification':
                y_ctx = torch.from_numpy(context_y[ctx]).long().unsqueeze(0)
                y_query = torch.from_numpy(context_y[query]).long()
                logits = net(x_ctx, y_ctx, x_query, mask, task_type='classification')
                loss = F.cross_entropy(logits[0, :, :self.model_._n_classes], y_query)
            else:
                y_ctx = torch.from_numpy(context_y[ctx]).float().unsqueeze(0)
                y_query = torch.from_numpy(context_y[query]).float().unsqueeze(0)
                y_ctx, y_query, _, _ = standardize_y_per_task(y_ctx, y_query)
                logits = net(x_ctx, y_ctx, x_query, mask, task_type='regression')
                loss = bar_distribution_loss(logits, y_query, net.reg_head)
            if not torch.isfinite(loss):
                raise ValueError('Hugging Face fine-tuning loss is not finite.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            self.loss_curve_.append({'step': step + 1, 'loss': float(loss.detach())})
        net.eval()
        return self

    def predict(self, X):
        if self.selector_ is not None:
            X = self.selector_.transform(X)
        X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
        return np.concatenate([self.model_.predict(X[i:i + 32]) for i in range(0, len(X), 32)])
