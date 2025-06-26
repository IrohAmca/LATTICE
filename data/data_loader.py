from collections import Counter
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset

try:
    spacy_en = spacy.load("en_core_web_sm")
    spacy_de = spacy.load("de_core_news_sm")
except OSError:
    print("Spacy models not found. Please install:")
    print("python -m spacy download en_core_web_sm")
    print("python -m spacy download de_core_news_sm")
    raise


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


BOS_WORD = "<s>"
EOS_WORD = "</s>"
BLANK_WORD = "<blank>"
UNK_WORD = "<unk>"


class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {
            BLANK_WORD: 0,
            UNK_WORD: 1,
            BOS_WORD: 2,
            EOS_WORD: 3,
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_count = Counter()

    def add_sentence(self, sentence):
        for word in sentence:
            self.word_count[word] += 1

    def build_vocab(self):
        for word, count in self.word_count.items():
            if count >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def __len__(self):
        return len(self.word2idx)

    def encode(self, sentence):
        return [self.word2idx.get(word, self.word2idx[UNK_WORD]) for word in sentence]

    def decode(self, indices):
        return [self.idx2word.get(idx, UNK_WORD) for idx in indices]


class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, max_len=100):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

        self.filtered_data = []
        for item in data:
            src_tokens = tokenize_de(item["de"])
            tgt_tokens = tokenize_en(item["en"])
            if len(src_tokens) <= max_len and len(tgt_tokens) <= max_len:
                self.filtered_data.append(item)

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        item = self.filtered_data[idx]
        src_text = item["de"]
        tgt_text = item["en"]

        src_tokens = tokenize_de(src_text)
        tgt_tokens = tokenize_en(tgt_text)

        src_indices = self.src_vocab.encode(src_tokens)
        tgt_indices = (
            [self.tgt_vocab.word2idx[BOS_WORD]]
            + self.tgt_vocab.encode(tgt_tokens)
            + [self.tgt_vocab.word2idx[EOS_WORD]]
        )

        return torch.tensor(src_indices), torch.tensor(tgt_indices)


def load_translation_dataset(
    dataset_name="wmt14", language_pair="de-en", max_samples=None
):

    try:
        if dataset_name == "wmt14":
            dataset = load_dataset("wmt14", "de-en")

        elif dataset_name == "opus100":
            dataset = load_dataset("opus100", f"{language_pair}")

        elif dataset_name == "iwslt2017":
            dataset = load_dataset("iwslt2017", "iwslt2017-de-en")

        elif dataset_name == "multi30k":
            dataset = load_dataset("multi30k", "de-en")

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        def format_data(examples):
            if dataset_name == "wmt14":
                return {
                    "de": [ex["de"] for ex in examples["translation"]],
                    "en": [ex["en"] for ex in examples["translation"]],
                }
            elif dataset_name == "multi30k":
                return {"de": examples["de"], "en": examples["en"]}
            else:
                return {
                    "de": examples["de"]
                    if "de" in examples
                    else examples["translation"]["de"],
                    "en": examples["en"]
                    if "en" in examples
                    else examples["translation"]["en"],
                }

        train_data = dataset["train"].map(format_data, batched=True)

        if "validation" in dataset:
            val_data = dataset["validation"].map(format_data, batched=True)
        else:
            split_data = train_data.train_test_split(test_size=0.1)
            train_data = split_data["train"]
            val_data = split_data["test"]

        if "test" in dataset:
            test_data = dataset["test"].map(format_data, batched=True)
        else:
            split_val = val_data.train_test_split(test_size=0.5)
            val_data = split_val["train"]
            test_data = split_val["test"]

        if max_samples:
            train_data = train_data.select(range(min(max_samples, len(train_data))))
            val_data = val_data.select(range(min(max_samples // 10, len(val_data))))
            test_data = test_data.select(range(min(max_samples // 10, len(test_data))))

        print(f"Loaded {dataset_name} dataset:")
        print(f"  Train: {len(train_data)} samples")
        print(f"  Validation: {len(val_data)} samples")
        print(f"  Test: {len(test_data)} samples")

        return train_data, val_data, test_data

    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        print("Falling back to dummy data...")
        return load_dummy_data()


def load_dummy_data():
    dummy_data = []
    src_sentences = [
        "Hallo, wie geht es dir?",
        "Ich bin ein Student.",
        "Das Wetter ist heute schön.",
        "Wir gehen ins Kino.",
        "Kannst du mir helfen?",
        "Ich lerne Deutsch.",
        "Das Buch ist interessant.",
        "Wir fahren nach Berlin.",
    ] * 50

    tgt_sentences = [
        "Hello, how are you?",
        "I am a student.",
        "The weather is nice today.",
        "We are going to the cinema.",
        "Can you help me?",
        "I am learning German.",
        "The book is interesting.",
        "We are going to Berlin.",
    ] * 50

    for src, tgt in zip(src_sentences, tgt_sentences):
        dummy_data.append({"de": src, "en": tgt})

    train_size = int(0.8 * len(dummy_data))
    val_size = int(0.1 * len(dummy_data))

    return (
        dummy_data[:train_size],
        dummy_data[train_size : train_size + val_size],
        dummy_data[train_size + val_size :],
    )


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return src_batch, tgt_batch


def create_data_loaders(
    dataset_name="multi30k", batch_size=32, max_len=100, min_freq=2, max_samples=1000
):
    train_data, val_data, test_data = load_translation_dataset(
        dataset_name=dataset_name, max_samples=max_samples
    )

    print("Building vocabularies...")
    src_vocab = Vocabulary(min_freq=min_freq)
    tgt_vocab = Vocabulary(min_freq=min_freq)

    for item in train_data:
        src_vocab.add_sentence(tokenize_de(item["de"]))
        tgt_vocab.add_sentence(tokenize_en(item["en"]))

    src_vocab.build_vocab()
    tgt_vocab.build_vocab()

    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")

    train_dataset = TranslationDataset(train_data, src_vocab, tgt_vocab, max_len)
    val_dataset = TranslationDataset(val_data, src_vocab, tgt_vocab, max_len)
    test_dataset = TranslationDataset(test_data, src_vocab, tgt_vocab, max_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


if __name__ == "__main__":
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_data_loaders(
        dataset_name="multi30k",
        batch_size=16,
        max_samples=1000,  
    )

    print("\nTesting data loaders...")
    for batch_idx, (src, tgt) in enumerate(train_loader):
        print(f"Batch {batch_idx}: src shape {src.shape}, tgt shape {tgt.shape}")

        if batch_idx == 0:
            print("\nFirst example:")
            print(f"Source: {' '.join(src_vocab.decode(src[0].tolist()))}")
            print(f"Target: {' '.join(tgt_vocab.decode(tgt[0].tolist()))}")

        if batch_idx == 2:
            break
