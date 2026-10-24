"""Stage 5 - Fine-tune a lightweight transformer as a regressor (num_labels=1).

Target scaled by /CAP during training, predictions converted back and clipped."""
from utils import ensure_dirs, set_seed


def main() -> None:
    set_seed()
    ensure_dirs()
    # TODO:
    # 1. AutoModelForSequenceClassification(name, num_labels=1, problem_type='regression')
    # 2. Long-text strategy: head / head+tail / chunking (check dates survive)
    # 3. Early stopping on validation MAE; log lr, batch size, epochs, max_len, time, hardware
    # 4. Optional: Huber loss vs MSE
    raise NotImplementedError


if __name__ == "__main__":
    main()
