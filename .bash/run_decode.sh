data_dir=iwslt14.tokenized.de-en
source_lang=de
target_lang=en
metric=pretrained
path=/home/neil/nanmt-fairseq/fairseq_easy_extend/models/nat/checkpoints/$metric/checkpoint_best.pt
output_file="outputs/decode_${metric}.out"

# Define scoring methods
scoring_methods=("bleu") # bleu, wer, meteor, bert_score

# Redirect all output to $output_file (append if file exists)
exec >> "$output_file" 2>&1

# Loop through scoring methods
for scoring in "${scoring_methods[@]}"; do
    echo "Generating with objective %s and scoring %s\n" "$metric" "$scoring \n"
    python decode.py $data_dir --source-lang $source_lang --target-lang $target_lang --task translation_lev --iter-decode-max-iter 9 --gen-subset test --print-step --remove-bpe --tokenizer nltk --scoring $scoring --path $path
    echo "--------------------------------------------------\n\n"
done
