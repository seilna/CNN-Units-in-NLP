#!/bin/sh

PS3='Please enter the nubmer of the task: '
options=("en-fr-news" "en-de-news" "en-cs-news" "en-de-europarl" "dbpedia" "yelp" "ag")
select opt in "${options[@]}"
do
    case $opt in
        "en-fr-news")
            echo "you chose en-fr-news"
            task="en-fr-news"
            break
            ;;
        "en-de-news")
            echo "you chose en-de-news"
            task="en-de-news"
            break
            ;;
        "en-cs-news")
            echo "you chose en-cs-news"
            task="en-cs-news"
            break
            ;;
        "en-de-europarl")
            echo "you chose en-de-europarl"
            task="en-de-europarl"
            break
            ;;
        "dbpedia")
            echo "you chose dbpedia"
            task="dbpedia"
            break
            ;;
        "yelp")
            echo "you chose yelp"
            task="yelp"
            break
            ;;
        "ag")
            echo "you chose ag"
            task="ag"
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done

top_k=10
num_units=1024


if [[ $task == *"en-"* ]]; then
    problem=translation
    num_layers=15

    src=$(jq .\"$task\".src config.js)
    tgt=$(jq .\"$task\".tgt config.js)
    pt=$(jq .\"$task\".pretrain config.js)

    src="${src%\"}"
    src="${src#\"}"

    tgt="${tgt%\"}"
    tgt="${tgt#\"}"

    pt="${pt%\"}"
    pt="${pt#\"}"

    activ_dump_fname=activ_dump_translator.py
    align_fname=align_concepts_translator.py

else
    problem=classification
    num_layers=4

    src=$(jq .\"$task\".src config.js)
    pt=$(jq .\"$task\".pretrain config.js)

    src="${src%\"}"
    src="${src#\"}"

    activ_dump_fname=activ_dump_classifier.py
    align_fname=align_concepts_classifier.py
fi


cd ../code

# Step 1. Feed all training sentences and save their activation values
if [[ $task == *"en-"* ]]; then
    for bucket in 50 100 150 200
    do
        python $activ_dump_fname --model_path=$pt --source_file=$src --target_file=$tgt --bucket_size=$bucket
    done

else
    python $activ_dump_fname --resume_model=$pt --database_path=$src --model_name=$task --batch_size=512
fi


# Step 2. Retreival of Top-K Sentence per unit
for ((layer=0;layer<$num_layers;layer++));
do
    python top_activated_sentences.py --layer=$layer --top_k=$top_k --task=$problem --model_name=$task &
done

wait
# Step 3. Align `--num_align` concepts to each unit (`--num_units`: # of units to align concepts)
for ((layer=0;layer<$num_layers;layer++));
do
    python $align_fname --layer_index=$layer --task=$task --num_align=3 --num_units=$num_units
done


# Step 4. Visualize aligned concepts & Top activated sentence per unit
python visualize.py --task=$task --top_k=$top_k --num_align=3 --num_units=$num_units
