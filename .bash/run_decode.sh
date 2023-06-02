data_dir=iwslt14.tokenized.de-en
source_lang=de
target_lang=en
metric=bleu
path=/home/neil/nanmt-fairseq/fairseq_easy_extend/models/nat/checkpoints/$metric/checkpoint_129_220.pt
output_file="outputs/decode_${metric}.out"

# Define scoring methods
scoring_methods=("meteor" "wer" "bert_score")

# Redirect all output to $output_file (append if file exists)
exec >> "$output_file" 2>&1

# Loop through scoring methods
for scoring in "${scoring_methods[@]}"; do
    printf "Generating with objective %s and scoring %s\n" "$metric" "$scoring \n"
    python decode.py $data_dir --source-lang $source_lang --target-lang $target_lang --task translation_lev --iter-decode-max-iter 9 --gen-subset test --print-step --remove-bpe --tokenizer moses --scoring $scoring --path $path
    printf "--------------------------------------------------\n\n"
done
