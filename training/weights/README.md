# Training weights

The files in this directory are the small, task-specific weights required to
reproduce or inspect the final SEG010 training state. They are separate from
the public Qwen base model and SigLIP encoder, which are downloaded from their
public sources as described in the repository README.

| File | Role |
|---|---|
| `canonical_union/union_adapted_weights.safetensors` | exact zero-output union initialization for final training |
| `general_final_union_adapted_weights.safetensors` | final General specialist state pack |
| `agg_final_union_adapted_weights.safetensors` | final Aggregation specialist state pack |

SHA256:

```text
3442593eaa7602a308c276028bef2cc16ae0a4c0eaa87996efbf314baca467c3  canonical_union/union_adapted_weights.safetensors
b77f1627709032f80d7c105e877a45330551de7a88a5cbfb1ccf52e83d4ef7f7  general_final_union_adapted_weights.safetensors
724ec447dbba79ff902422f34d0ae14801e8de4d318be7f8474b45bd30a91739  agg_final_union_adapted_weights.safetensors
```

The recovery optimizer/RNG states and historical checkpoints are not part of
the public training package. They are not needed for final-model inference or
for reproducing the final recipe from its canonical initialization.
