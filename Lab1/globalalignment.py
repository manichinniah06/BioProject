import pandas as pd
from Bio.Align import PairwiseAligner

WT_P53 = (
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRM"
    "PEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNK"
    "MFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVE"
    "GNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNL"
    "LGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTL"
    "QIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
)

def perform_global_alignment(df):

    print("="*70)
    print("GLOBAL ALIGNMENT VALIDATION")
    print("="*70)

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = -1
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -10

    alignment_scores = []
    identities = []
    hamming_distances = []

    for seq in df['sequence']:

        score = aligner.score(WT_P53, seq)
        alignment_scores.append(score)

        mismatches = sum(1 for a, b in zip(WT_P53, seq) if a != b)
        hamming_distances.append(mismatches)

        identity = ((len(WT_P53) - mismatches) / len(WT_P53)) * 100
        identities.append(identity)

    df['alignment_score'] = alignment_scores
    df['percent_identity'] = identities
    df['hamming_distance'] = hamming_distances

    print(f"Sequence Length: {len(WT_P53)}")
    print(f"Min Alignment Score: {df['alignment_score'].min()}")
    print(f"Max Alignment Score: {df['alignment_score'].max()}")
    print(f"Mean Alignment Score: {df['alignment_score'].mean():.2f}")

    print(f"\nMin Percent Identity: {df['percent_identity'].min():.2f}%")
    print(f"Max Percent Identity: {df['percent_identity'].max():.2f}%")

    print("\nUnique Hamming Distances:", df['hamming_distance'].unique())

    if all(df['hamming_distance'] == 1):
        print("\n✔ All sequences contain exactly ONE amino acid mutation (Valid Missense Dataset)")
    else:
        print("\n⚠ Some sequences contain multiple mutations!")

    print("="*70)

    df.to_csv("p53_dataset_with_alignment_validation.csv", index=False)
    print("\nAligned dataset saved as: p53_dataset_with_alignment_validation.csv")

    return df

if __name__ == "__main__":

    df = pd.read_csv("p53_dataset_non_redundant.csv")
    perform_global_alignment(df)
