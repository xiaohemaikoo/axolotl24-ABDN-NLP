# axolotl24-ABDN-NLP

This package is designed to detect word sense changes and to generate definition for the new, gained senses of the target word,
which is also designed to the task of the [AXOLOTL-24 Shared Task](https://github.com/ltgoslo/axolotl24_shared_task).


## Track 1

**Bridging diachronic word uses and a synchronic dictionary:** This task is to identify mappings between dictionary entries and the word usages of each target word, i.e., that the task asks to detect whether each word usage has a novel sense or not, meaning that it is not (or is) recorded in dictionaries.

### Running dynamic sense cluster to detect word sense changes for track 1

```commandline
python3 track1_dym_sense.py --test ../../data/russian/axolotl.dev.ru.tsv --pred pred_dev_ru_test.tsv --lang fi
```


### Evaluating results for track 1

```commandline
python3 scorer_track1.py --gold ../../data/axolotl.ru/axolotl.dev.ru.tsv --pred ../baselines/pred_dev_ru.tsv
```

## Track 2

**Definition generation for novel word senses:**  This task builds upon the mapping results of Subtask 1. It aims to generate dictionary-like definitions for the unmatched word usages discovered in Subtask 1, i.e., that these usages contain novel senses not covered by dictionaries.

### Running script to generate novel sense definition via GPT for track 2

```commandline
python3 track2_gpt.py --test_data ../../data/axolotl.fi/axolotl.dev.fi.tsv --predictions_file ../baselines/pred_dev_fi.track2.tsv --lang fi
```
### Or running script to generate novel sense definition via LLAMA for track 2

```commandline
python3 track2_llama.py --test_data ../../data/axolotl.fi/axolotl.dev.fi.tsv --predictions_file ../baselines/pred_dev_fi.track2.tsv --lang fi
```

### Evaluating results for track 2

```commandline
python3 scorer_track2.py ../baselines/pred_dev_fi.track2.tsv ../../data/axolotl.fi/axolotl.dev.fi.tsv scores.fi.txt
```