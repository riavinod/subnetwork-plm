from esm.pretrained import load_model_and_alphabet
from transformers import AutoTokenizer, EsmForSequenceClassification


def print_available_models():
    # List of available ESM models for proteins
    protein_models = [
        # "esm1b_t33_650M_UR50S",
        # "esm1b_t33_650M_UR50D",
        # "esm1b_t33_650M_UR100",
        # "esm1b_t33_650M_UR50S_small",
        # "esm1_t34_670M_UR50S",
        # "esm1_t6_43M_UR50S",
        # "esm1v_t33_650M_UR50S",
        # "esm1v_t33_650M_UR50D",
        # "esm1v_t33_650M_UR100",
        # "esm1v_t33_650M_UR50S_small",
        # "esm1v_t6_43M_UR50S",
        # "esm1v_t6_43M_UR50D",
        # "esm1v_t6_43M_UR100",
        "esm1v_t6_43M_UR50S_small"
    ]
    
    model_name = 'esm2_t6_8M_UR50D'
    model = EsmForSequenceClassification.from_pretrained("facebook/esm2_t6_8M_UR50D")
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name}: Size - {num_parameters / 1e6} million parameters")

    print(dir(model))
if __name__ == "__main__":
    print_available_models()
