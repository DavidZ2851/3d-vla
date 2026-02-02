import torch
from torch import nn
from torch.nn import functional as F
# from open_clip import create_model_from_pretrained, get_tokenizer


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


def text_global_pool(x, text=None, pool_type='argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


class SigLip2TextEncoder(nn.Module):

    def __init__(self, model_id='hf-hub:timm/ViT-B-16-SigLIP2-512'):
        super().__init__()
        self.model, self.preprocess = create_model_from_pretrained(model_id)
        self.tokenizer = get_tokenizer(model_id)
        del self.model.visual
        self.model.eval()

    def forward(self, text, normalize=False):
        cast_dtype = self.model.text.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.model.text.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.model.text.attn_mask
        if self.model.text.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, _expand_token(self.model.text.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.model.text.build_cls_mask(text, cast_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.model.text.positional_embedding[:seq_len].to(cast_dtype)
        x = self.model.text.transformer(x, attn_mask=attn_mask)

        # x.shape = [batch_size, n_ctx, transformer.width]
        if self.model.text.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled, tokens = text_global_pool(x, pool_type='last')
            pooled = self.model.text.ln_final(pooled)  # final LN applied after pooling in this case
        else:
            x = self.model.text.ln_final(x)
            pooled, tokens = text_global_pool(x, text, pool_type=self.model.text.pool_type)

        if self.model.text.text_projection is not None:
            if isinstance(self.model.text.text_projection, nn.Linear):
                pooled = self.model.text.text_projection(pooled)
            else:
                pooled = pooled @ self.model.text.text_projection

        return F.normalize(pooled, dim=-1) if normalize else tokens
