import pytest
import torch
from automl_framework.model.architecture.transformer_encoder import TransformerBasedRegression

def test_transformer_regression_scalar():
    """Verify that when predict_distribution=False, the output is a single scalar per sample."""
    batch_size = 4
    seq_len = 10
    input_dim = 16
    
    model = TransformerBasedRegression(
        input_dim=input_dim,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        predict_distribution=False
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, 1)
    
    # Check that gradients can flow back
    loss = output.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient!"


def test_transformer_regression_distribution():
    """Verify that when predict_distribution=True, the outputs are mean and positive variance."""
    batch_size = 4
    seq_len = 10
    input_dim = 16
    
    model = TransformerBasedRegression(
        input_dim=input_dim,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        predict_distribution=True
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    mean, variance = model(x)
    
    # Check output shapes
    assert mean.shape == (batch_size, 1)
    assert variance.shape == (batch_size, 1)
    
    # Check variance positivity
    assert torch.all(variance > 0.0)
    
    # Check that gradients can flow back
    loss = (mean.sum() + variance.sum())
    loss.backward()
    
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient!"


@pytest.mark.parametrize("pooling", ["mean", "max", "last"])
def test_pooling_strategies(pooling):
    """Verify that all pooling options run and return the correct shape."""
    batch_size = 3
    seq_len = 8
    input_dim = 8
    
    model = TransformerBasedRegression(
        input_dim=input_dim,
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        pooling=pooling,
        predict_distribution=False
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    
    assert output.shape == (batch_size, 1)


def test_masking_support():
    """Verify that sequence padding masks work correctly."""
    batch_size = 2
    seq_len = 5
    input_dim = 8
    
    model = TransformerBasedRegression(
        input_dim=input_dim,
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        pooling="mean",
        predict_distribution=False
    )
    
    x = torch.randn(batch_size, seq_len, input_dim)
    # Mask out the last two tokens of the second sequence
    mask = torch.tensor([
        [False, False, False, False, False],
        [False, False, False, True, True]
    ])
    
    output = model(x, mask=mask)
    assert output.shape == (batch_size, 1)
    
    # Check gradients flow back
    loss = output.sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None


def test_transformer_wrapper_standard():
    """Verify that ModelWrapperTransformer works on standard pandas structures in scalar mode."""
    import pandas as pd
    import numpy as np
    from automl_framework.model.wrappers import ModelWrapperTransformer
    
    batch_size = 10
    num_features = 5
    
    # 2D tabular features
    X = pd.DataFrame(np.random.randn(batch_size, num_features), columns=[f"F{i}" for i in range(num_features)])
    y = pd.Series(np.random.randn(batch_size))
    
    # input_dim=1 since we treat columns as sequence steps
    model = TransformerBasedRegression(
        input_dim=1,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        predict_distribution=False
    )
    
    wrapper = ModelWrapperTransformer(
        name="TransformerTest",
        model_instance=model,
        epochs=5,
        lr=0.01,
        batch_size=4,
        verbose=False
    )
    
    # Train
    wrapper.fit(X, y)
    
    # Predict
    preds = wrapper.predict(X)
    assert preds.shape == (batch_size,)
    assert isinstance(preds, np.ndarray)


def test_transformer_wrapper_distribution():
    """Verify that ModelWrapperTransformer works in probabilistic mode."""
    import pandas as pd
    import numpy as np
    from automl_framework.model.wrappers import ModelWrapperTransformer
    
    batch_size = 8
    num_features = 4
    
    X = pd.DataFrame(np.random.randn(batch_size, num_features), columns=[f"F{i}" for i in range(num_features)])
    y = pd.Series(np.random.randn(batch_size))
    
    model = TransformerBasedRegression(
        input_dim=1,
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        predict_distribution=True
    )
    
    wrapper = ModelWrapperTransformer(
        name="TransformerProbTest",
        model_instance=model,
        epochs=5,
        lr=0.01,
        batch_size=4,
        verbose=False
    )
    
    wrapper.fit(X, y)
    
    preds = wrapper.predict(X)
    assert preds.shape == (batch_size,)


def test_transformer_model_pool_integration():
    """Verify that the Transformer builder registers and instantiates via ModelPool correctly."""
    from automl_framework.model.model_pool import ModelPool
    from automl_framework.model.wrappers import ModelWrapperTransformer
    
    config = {
        "framework": {
            "active_models": ["Transformer"]
        },
        "models": {
            "Transformer": {
                "epochs": 2,
                "lr": 0.005,
                "batch_size": 2,
                "input_dim": 1,
                "d_model": 16,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 32,
                "predict_distribution": False
            }
        }
    }
    
    pool = ModelPool(random_state=42, config=config)
    assert "Transformer" in pool.list_available_models()
    
    transformer_wrap = pool.get_model("Transformer")
    assert isinstance(transformer_wrap, ModelWrapperTransformer)
    assert transformer_wrap.epochs == 2
    assert transformer_wrap.lr == 0.005
    assert transformer_wrap.batch_size == 2

