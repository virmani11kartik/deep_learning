from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers, processors
import os
import sys

def main():
    vocab_s = 2048
    min_freq = 2
    special_tok = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    shakespeare_path = os.path.join(data_dir, 'shakespeare.txt')
    baskerville_path = os.path.join(data_dir, 'hounds_of_baskervilles.txt')  # ensure filename is correct
    war_and_peace_path = os.path.join(data_dir, 'war_and_peace.txt')

    # Verify files exist
    for p in [shakespeare_path, baskerville_path, war_and_peace_path]:
        if not os.path.isfile(p):
            print(f"Missing file: {p}", file=sys.stderr)
            sys.exit(1)

    # MODEL
    bpe_model = models.BPE(unk_token="[UNK]")
    token = Tokenizer(bpe_model)

    # NORMALIZATION
    token.normalizer = normalizers.BertNormalizer(
        clean_text = True,
        handle_chinese_chars= True,
        strip_accents = True,
        lowercase=True
    )

    # PRE_TOKENIZER
    token.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    #POST-PROCESS
    token.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[
            ("[CLS]", 0),   
            ("[SEP]", 0),
        ],
    )

    # TRAIN
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_s,
        min_frequency= min_freq,
        special_tokens=special_tok
    )

    token.train(
        files=[shakespeare_path, baskerville_path, war_and_peace_path],
        trainer=trainer
    )

    pad_id = token.token_to_id("[PAD]")
    cls_id = token.token_to_id("[CLS]")
    sep_id = token.token_to_id("[SEP]")

    token.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[
            ("[CLS]", cls_id),
            ("[SEP]", sep_id),
        ],
    )

    token.enable_padding(direction="right", pad_id=pad_id, pad_token="[PAD]")
    token.enable_truncation(max_length=128)

    token.save("tokenizer.json")
    print("Saved tokenizer to json")

if __name__ == "__main__":
    main()