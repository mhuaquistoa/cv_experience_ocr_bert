"""Stage 2b - Reading order, cleaning, PII removal and masking of direct answers.

Output: data/processed/cvs.csv (id, occupation, text, exp_year) + masking report."""
from utils import ensure_dirs, set_seed


def main() -> None:
    set_seed()
    ensure_dirs()
    # TODO:
    # 1. Rebuild reading order from boxes (top-to-bottom, left-to-right, handle 2 columns)
    # 2. Fix broken lines / hyphenation; optional low-confidence filter (justify threshold)
    # 3. Remove name, email, phone, address
    # 4. Mask 'N years of experience', age, date of birth; count affected CVs
    #    (job dates and graduation years must stay)
    raise NotImplementedError


if __name__ == "__main__":
    main()
