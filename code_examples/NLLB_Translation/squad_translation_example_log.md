# Translation of SQUAD with NLLB

## Data Download Link

https://huggingface.co/datasets/rajpurkar/squad/tree/main/plain_text  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ ls
parquet_extractor.py  train-00000-of-00001.parquet  validation-00000-of-00001.parquet
```

## Code Info

https://github.com/ye-kyaw-thu/tools/blob/master/python/parquet_extractor.py  
  
## Extraction  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ python ./parquet_extractor.py --parquet_file ./validation-00000-of-00001.parquet
Parquet file './validation-00000-of-00001.parquet' extracted to 'validation-00000-of-00001.csv'
```

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ python ./parquet_extractor.py --parquet_file ./train-00000-of-00001.parquet
Parquet file './train-00000-of-00001.parquet' extracted to 'train-00000-of-00001.csv'
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$
```

## Check  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ wc *.csv
   94670 11915768 84323576 train-00000-of-00001.csv
   19477  1575940 11146209 validation-00000-of-00001.csv
  114147 13491708 95469785 total
  
```

vi editor နဲ့ ကြည့်ရင် ...  

```
    1 id,title,context,question,answers
    2 5733be284776f41900661182,University_of_Notre_Dame,"Architecturally, the school has a Catholic chara      cter. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in fron      t of the Main Building and facing it, is a copper statue of Christ with arms upraised with the lege      nd ""Venite Ad Me Omnes"". Next to the Main Building is the Basilica of the Sacred Heart. Immediate      ly behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of t      he grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous       in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and th      e Gold Dome), is a simple, modern stone statue of Mary.",To whom did the Virgin Mary allegedly appe      ar in 1858 in Lourdes France?,"{'text': array(['Saint Bernadette Soubirous'], dtype=object), 'answe      r_start': array([515], dtype=int32)}"
    3 5733be284776f4190066117f,University_of_Notre_Dame,"Architecturally, the school has a Catholic chara      cter. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in fron      t of the Main Building and facing it, is a copper statue of Christ with arms upraised with the lege      nd ""Venite Ad Me Omnes"". Next to the Main Building is the Basilica of the Sacred Heart. Immediate      ly behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of t      he grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous       in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and th      e Gold Dome), is a simple, modern stone statue of Mary.",What is in front of the Notre Dame Main Bu      ilding?,"{'text': array(['a copper statue of Christ'], dtype=object), 'answer_start': array([188],       dtype=int32)}"
	
```

## Coding

```python
#!/usr/bin/env python3
import argparse
import csv
import sys
import os

def extract_field(input_file, column, output_file=None):
    try:
        with open(input_file, 'r', encoding='utf-8') as csvfile:
            # Use csv reader to properly handle quoted fields with commas
            reader = csv.DictReader(csvfile)
            
            if column not in reader.fieldnames:
                print(f"Error: Column '{column}' not found in CSV file. Available columns: {', '.join(reader.fieldnames)}")
                sys.exit(1)
                
            data = [row[column] for row in reader]
            
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as outfile:
                    for item in data:
                        outfile.write(item + '\n')
                print(f"Successfully extracted '{column}' to {output_file}")
            else:
                for item in data:
                    print(item)
                    
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Extract specific fields from SQUAD CSV files')
    parser.add_argument('--input', required=True, help='Input CSV filename')
    parser.add_argument('--column', required=True, help='Column name to extract')
    parser.add_argument('--output', help='Output filename (optional)')
    
    args = parser.parse_args()
    
    extract_field(args.input, args.column, args.output)

if __name__ == '__main__':
    main()
```

Batch Code...  

```bash
#!/bin/bash

# Function to process a single CSV file
process_file() {
    local input_file=$1
    local prefix=$2
    
    echo "Processing $input_file..."
    
    # Get all columns from the CSV file
    columns=$(python3 -c "
import csv
with open('$input_file', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    print(' '.join(reader.fieldnames))
")
    
    # Extract each column
    for col in $columns; do
        output_file="${prefix}_${col}.txt"
        python3 extract_csv_field.py --input "$input_file" --column "$col" --output "$output_file"
    done
}

# Main script
echo "Starting SQUAD dataset extraction..."

# Process training file
process_file "train-00000-of-00001.csv" "train"

# Process validation file
process_file "validation-00000-of-00001.csv" "valid"

echo "Extraction completed successfully."
```

Run လို့ ရဖို့အတွက်က executable mode အဖြစ် chmod Linux commandline tool နဲ့ ပြောင်းပေးရတယ်။  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ nano extract_squad_fields.sh
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ (base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ chmod +x ./extract_squad_fields.sh
```

လက်ရှိ အလုပ်လုပ်နေတဲ့ folder အောက်မှာ ရှိနေတဲ့ ဖိုင်တွေက အောက်ပါအတိုင်း...  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ ls
extract_csv_field.py     train-00000-of-00001.csv       validation-00000-of-00001.parquet
extract_squad_fields.sh  train-00000-of-00001.parquet
parquet_extractor.py     validation-00000-of-00001.csv
```

## Extracting All Fields  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ time ./extract_squad_fields.sh
Starting SQUAD dataset extraction...
Processing train-00000-of-00001.csv...
Successfully extracted 'id' to train_id.txt
Successfully extracted 'title' to train_title.txt
Successfully extracted 'context' to train_context.txt
Successfully extracted 'question' to train_question.txt
Successfully extracted 'answers' to train_answers.txt
Processing validation-00000-of-00001.csv...
Successfully extracted 'id' to valid_id.txt
Successfully extracted 'title' to valid_title.txt
Successfully extracted 'context' to valid_context.txt
Successfully extracted 'question' to valid_question.txt
Successfully extracted 'answers' to valid_answers.txt
Extraction completed successfully.

real    0m7.598s
user    0m7.216s
sys     0m0.383s
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$
```

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ ls *.txt
train_answers.txt  train_id.txt        train_title.txt    valid_context.txt  valid_question.txt
train_context.txt  train_question.txt  valid_answers.txt  valid_id.txt       valid_title.txt
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$
```

Checking train files...  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ wc train_*.txt
   94140   714994  8706546 train_answers.txt
   88128 10491130 66304332 train_context.txt
   87599    87599  2189975 train_id.txt
   87599   881343  5307142 train_question.txt
   87599    87599  1322786 train_title.txt
  445065 12262665 83830781 total
```

original CSV ဖိုင်နဲ့ နှိုင်းယှဉ်ကြည့်တဲ့အခါမှာ...  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ wc train-00000-of-00001.csv
   94670 11915768 84323576 train-00000-of-00001.csv
```

For validation files...  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ wc valid_*.txt
   19092   178509  1754723 valid_answers.txt
   10954  1310201  8258955 valid_context.txt
   10570    10570   264250 valid_id.txt
   10570   107990   645024 valid_question.txt
   10570    10570   163387 valid_title.txt
   61756  1617840 11086339 total
```

original csv ဖိုင်နဲ့ နှိုင်းယှဉ်ကြည့်တဲ့အခါမှာ...  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ wc validation-00000-of-00001.csv
   19477  1575940 11146209 validation-00000-of-00001.csv
```

အထက်မှာ တွေ့ရတဲ့အတိုင်းပဲ id, question, title က လိုင်းအရေအတွက် တူပေမဲ့ answers, context မှာ လိုင်းအရေအတွက်က မညီဘူး။ ပြီးတော့ original csv ဖိုင်နဲ့လည်း တစ်ဖိုင်မှ လိုင်းအရေအတွက် မညီဘူး (header line တစ်လိုင်းကွာတာမျိုးမဟုတ်ပဲ) ... ပြန်စစ်ဖို့ လိုအပ်။  

Zip ဖိုင်လုပ်...  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data$ zip -r squad.zip ./squad
  adding: squad/ (stored 0%)
  adding: squad/train_question.txt (deflated 68%)
  adding: squad/valid_title.txt (deflated 99%)
  adding: squad/valid_question.txt (deflated 69%)
  adding: squad/valid_answers.txt (deflated 84%)
  adding: squad/extract_csv_field.py (deflated 60%)
  adding: squad/validation-00000-of-00001.csv (deflated 89%)
  adding: squad/validation-00000-of-00001.parquet (deflated 15%)
  adding: squad/train_context.txt (deflated 90%)
  adding: squad/extract_squad_fields.sh (deflated 48%)
  adding: squad/train_answers.txt (deflated 86%)
  adding: squad/valid_context.txt (deflated 91%)
  adding: squad/train_title.txt (deflated 99%)
  adding: squad/parquet_extractor.py (deflated 50%)
  adding: squad/train_id.txt (deflated 87%)
  adding: squad/valid_id.txt (deflated 87%)
  adding: squad/train-00000-of-00001.csv (deflated 89%)
  adding: squad/train-00000-of-00001.parquet (deflated 16%)
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data$
```

ကိုယ့် local စက်ထဲကို ကော်ပီကူး၊ manual စစ်ခဲ့...  

ဘာသွားတွေ့သလဲ လို့မေးရင် original csv ဖိုင်မှာကတည်းက field တစ်ခုအတွင်းမှာ enter ခေါက်ထားတာတွေကို တွေ့ရတယ်။ ဥပမာ အောက်ပါအတိုင်း...  

```
56be4db0acb8001400a502ed,Super_Bowl_50,"Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the ""golden anniversary"" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as ""Super Bowl L""), so that the logo could prominently feature the Arabic numerals 50.",Which NFL team represented the NFC at Super Bowl 50?,"{'text': array(['Carolina Panthers', 'Carolina Panthers', 'Carolina Panthers'],
      dtype=object), 'answer_start': array([249, 249, 249], dtype=int32)}"
```

```
{'text': array(['slug', 'metric slug', 'metric slug', 'metric slug',
       'the metric slug'], dtype=object), 'answer_start': array([274, 267, 267, 267, 263], dtype=int32)}
```

```
The problem is that some fields in the CSV contain newline characters within them, which causes the line counts to mismatch when we extract them. We need to ensure each record is written as a single line in the output files, even if the original field contained newlines.
```

## Python Code Updating  

clean_text() function ဖြည့်ရေးခဲ့တယ်။  

```python
#!/usr/bin/env python3
import argparse
import csv
import sys
import os

def clean_text(text):
    """Remove newlines and extra spaces from text"""
    if not text:
        return text
    # Replace newlines with spaces
    text = text.replace('\r', ' ').replace('\n', ' ')
    # Collapse multiple spaces into one
    return ' '.join(text.split())

def extract_field(input_file, column, output_file=None):
    try:
        with open(input_file, 'r', encoding='utf-8') as csvfile:
            # Use csv reader to properly handle quoted fields with commas
            reader = csv.DictReader(csvfile)
            
            if column not in reader.fieldnames:
                print(f"Error: Column '{column}' not found in CSV file. Available columns: {', '.join(reader.fieldnames)}")
                sys.exit(1)
                
            data = []
            for row in reader:
                field_value = row[column]
                # Clean the text by removing internal newlines
                cleaned_value = clean_text(field_value)
                data.append(cleaned_value)
            
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as outfile:
                    for item in data:
                        outfile.write(item + '\n')
                print(f"Successfully extracted '{column}' to {output_file}")
            else:
                for item in data:
                    print(item)
                    
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Extract specific fields from SQUAD CSV files')
    parser.add_argument('--input', required=True, help='Input CSV filename')
    parser.add_argument('--column', required=True, help='Column name to extract')
    parser.add_argument('--output', help='Output filename (optional)')
    
    args = parser.parse_args()
    
    extract_field(args.input, args.column, args.output)

if __name__ == '__main__':
    main()
```

## Redo Field Extractions 

ဒီတခါတော့ updated python code နဲ့ နောက်တခေါက် ထပ် field extraction လုပ်ကြည့်မယ်။  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ rm *.txt
```

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ time ./extract_squad_fields.sh
Starting SQUAD dataset extraction...
Processing train-00000-of-00001.csv...
Successfully extracted 'id' to train_id.txt
Successfully extracted 'title' to train_title.txt
Successfully extracted 'context' to train_context.txt
Successfully extracted 'question' to train_question.txt
Successfully extracted 'answers' to train_answers.txt
Processing validation-00000-of-00001.csv...
Successfully extracted 'id' to valid_id.txt
Successfully extracted 'title' to valid_title.txt
Successfully extracted 'context' to valid_context.txt
Successfully extracted 'question' to valid_question.txt
Successfully extracted 'answers' to valid_answers.txt
Extraction completed successfully.

real    0m8.982s
user    0m8.507s
sys     0m0.477s
```

no. of lines ကို ပြန်စစ်ကြည့်ခဲ့...  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ wc -l valid_*.txt
   10570 valid_answers.txt
   10570 valid_context.txt
   10570 valid_id.txt
   10570 valid_question.txt
   10570 valid_title.txt
   52850 total
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ wc -l validation-00000-of-00001.csv
19477 validation-00000-of-00001.csv
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$
```

training ဖိုင်အတွက်လည်း no. of line အိုကေရဲ့လား ထပ်စစ်ကြည့်ခဲ့...  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ wc -l train_*.txt
   87599 train_answers.txt
   87599 train_context.txt
   87599 train_id.txt
   87599 train_question.txt
   87599 train_title.txt
  437995 total
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ wc -l train-00000-of-00001.csv
94670 train-00000-of-00001.csv
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$
```

လိုင်းမညီတဲ့ ပြဿနာတော့ အဆင်ပြေသွားပြီ။ :)  

လက်ရှိ preprocessed လုပ်ထားတဲ့ SQUAD data path က အောက်ပါအတိုင်း...  

```
/home/ye/ye/exp/gpt-mt/nllb/data/squad
```

## Preparation for NLLB Translation  

ID ဖိုင်၊ title ဖိုင်တွေက translation မလုပ်သင့်ဘူး။  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ head ./valid_title.txt
Super_Bowl_50
Super_Bowl_50
Super_Bowl_50
Super_Bowl_50
Super_Bowl_50
Super_Bowl_50
Super_Bowl_50
Super_Bowl_50
Super_Bowl_50
Super_Bowl_50
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ tail ./valid_title.txt
Force
Force
Force
Force
Force
Force
Force
Force
Force
Force
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$
```

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ head ./train_title.txt
University_of_Notre_Dame
University_of_Notre_Dame
University_of_Notre_Dame
University_of_Notre_Dame
University_of_Notre_Dame
University_of_Notre_Dame
University_of_Notre_Dame
University_of_Notre_Dame
University_of_Notre_Dame
University_of_Notre_Dame
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ tail ./train_title.txt
Kathmandu
Kathmandu
Kathmandu
Kathmandu
Kathmandu
Kathmandu
Kathmandu
Kathmandu
Kathmandu
Kathmandu
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$
```

translation အဓိက လုပ်ရမှာက context ရယ်၊ question ရယ်ပဲ။  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ head -n 3 ./train_context.txt | nl
     1  Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.
     2  Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.
     3  Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$
```

question ဖိုင် ဥပမာ ...  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ head -n 5 ./train_question.txt | nl
     1  To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?
     2  What is in front of the Notre Dame Main Building?
     3  The Basilica of the Sacred heart at Notre Dame is beside to which structure?
     4  What is the Grotto at Notre Dame?
     5  What sits on top of the Main Building at Notre Dame?
```

answer ဖိုင်ကတော့ format က အောက်ပါအတိုင်းမို့လို့ မြန်မာစာအတွက်က သပ်သပ် ထပ်ပြင်ရလိမ့်မယ်။  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ head -n 5 ./train_answers.txt | nl
     1  {'text': array(['Saint Bernadette Soubirous'], dtype=object), 'answer_start': array([515], dtype=int32)}
     2  {'text': array(['a copper statue of Christ'], dtype=object), 'answer_start': array([188], dtype=int32)}
     3  {'text': array(['the Main Building'], dtype=object), 'answer_start': array([279], dtype=int32)}
     4  {'text': array(['a Marian place of prayer and reflection'], dtype=object), 'answer_start': array([381], dtype=int32)}
     5  {'text': array(['a golden statue of the Virgin Mary'], dtype=object), 'answer_start': array([92], dtype=int32)}
```

## Coding for Training Context Translation

folder တစ်ခုအောက်ထဲမှာထားပြီး အဲဒီထဲက txt ဖိုင်အားလုံးကို ဘာသာပြန်ခိုင်းတဲ့ ပုံစံနဲ့ သွားချင်လို့ ./train, ./valid ဆိုတဲ့ ဖိုလ်ဒါနှစ်ခုဆောက်ပြီး အောက်ပါအတိုင်း ပြင်ဆင်ခဲ့...  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ mkdir train
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ cp train_context.txt ./train/
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ cp train_question.txt ./train/
```

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ mkdir valid
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ cp valid_context.txt ./valid/
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ cp valid_question.txt ./valid
```

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ ls ./train
train_context.txt  train_question.txt
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/data/squad$ ls ./valid
valid_context.txt  valid_question.txt
```

ပြင်ဆင်ခဲ့တဲ့ shell script (squad_train2my.sh) က အောက်ပါအတိုင်း...  

```bash
#!/bin/bash

# Base directory for input files
INPUT_DIR="/home/ye/ye/exp/gpt-mt/nllb/data/squad/train/"

# Directory for output files
OUTPUT_DIR="/home/ye/ye/exp/gpt-mt/nllb/squad-my/"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Iterate over each .src file in the input directory
for FILE in "$INPUT_DIR"/*.txt; do
    # Extract the base filename without the extension
    BASENAME=$(basename "$FILE" .txt)

    # Define the output file name
    OUTPUT_FILE="$OUTPUT_DIR/$BASENAME.my"

    # Print the command being executed (for debugging)
    echo "Running nllb-translate.sh for $FILE"

    # Run the translation command
    time ./nllb-translate.sh --input "$FILE" --source eng_Latn --target mya_Mymr --output "$OUTPUT_FILE"
done
```

translation process က အနည်းဆုံး တပတ်လောက် ကြာနိုင်တာမို့လို့ server ကို ချိတ်ထားတဲ့ terminal ရဲ့ connection က ပြတ်သွားရင် program က ရပ်သွားနိုင်တာမို့လို့ screen command ကို သုံးကြရအောင်။  
screen command ကို သုံးတတ်မှ ဖြစ်မယ်။  

```
screen -S squad
```

Start translation ...  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb$ ./squad_train2my.sh | tee squad_train_tran.log
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb$ ./squad_train2my.sh | tee squad_train_tran.log
Running nllb-translate.sh for /home/ye/ye/exp/gpt-mt/nllb/data/squad/train//train_context.txt
JSON Payload: {
  "text": "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"ဗိသုကာအရ ကျောင်းသည် ကက်သလစ်ဘာသာဝင်ဖြစ်ပါသည်။ ဗိမာန်အကြီးဆဆုံး၏ ရွှေဂူပလာတွင် ဗနုမာရီဘုရား၏ ရွှေရောင်ရုပ်တုတစ်ခုရှိသည်။ ဗာနာနုမဟာမိတ်တော်၏ ရှေ့တွင် ဗားနုမေရီဘုရားရှိရာ၏ လက်မောင်းများဖြင့် ခရစ်တော်၏ ကြေးနီရုပ်တုတစ်ရုပ်ရှိသည်။"}
JSON Payload: {
  "text": "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
...
...
...
JSON Payload: {
  "text": "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend \"Venite Ad Me Omnes\". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"ဗိသုကာအရ ကျောင်းသည် ကက်သလစ်ဘာသာဝင်ဖြစ်ပါသည်။ ဗိမာန်အကြီးဆဆုံး၏ ရွှေဂူပလာတွင် ဗနုမာရီဘုရား၏ ရွှေရောင်ရုပ်တုတစ်ခုရှိသည်။ ဗာနာနုမဟာမိတ်တော်၏ ရှေ့တွင် ဗားနုမေရီဘုရားရှိရာ၏ လက်မောင်းများဖြင့် ခရစ်တော်၏ ကြေးနီရုပ်တုတစ်ရုပ်ရှိသည်။"}
...
...
JSON Payload: {
  "text": "As at most other universities, Notre Dame's students run a number of news media outlets. The nine student-run outlets include three newspapers, both a radio and television station, and several magazines and journals. Begun as a one-page journal in September 1876, the Scholastic magazine is issued twice monthly and claims to be the oldest continuous collegiate publication in the United States. The other magazine, The Juggler, is released twice a year and focuses on student literature and artwork. The Dome yearbook is published annually. The newspapers have varying publication interests, with The Observer published daily and mainly reporting university and other news, and staffed by students from both Notre Dame and Saint Mary's College. Unlike Scholastic and The Dome, The Observer is an independent publication and does not have a faculty advisor or any editorial oversight from the University. In 1987, when some students believed that The Observer began to show a conservative bias, a liberal newspaper, Common Sense was published. Likewise, in 2003, when other students believed that the paper showed a liberal bias, the conservative paper Irish Rover went into production. Neither paper is published as often as The Observer; however, all three are distributed to all students. Finally, in Spring 2008 an undergraduate journal for political science research, Beyond Politics, made its debut.",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"Notre Dame ၏ကျောင်းသားများသည် သတင်းမီဒီယာအတော်များများကကို လည်ပတ်သည်။ ကျောင်းသားများက လည်ပတ်သော ၉ ခုသော သတင်းဌာနများတွင် ရေဒီယယိုနှင့် ရုပ်မြင်သံကြားဌာနတစ်ခုနှင့် မဂ္ဂဇင်းနှင့် ဂျာနယ်များစွာပါဝင်သည်။ ၁၈၇၆ ခုနှစ် စက်တင်ဘာလတွင် စာမျက်နှာတစ်မျက်နှာစာ ဂျာနလအဖြစ်စတင်ထုတ်ဝေခခဲ့ပြီး အမေရိကန်ပြည်ထောင်စုတွင် ရှေးကျဆဆုံးဆက်တတိုက် ကောလိပ်ထုတ်ဝေမှုဖြစ်သည်ဟုဆဆိုကာ လစဉ်နှစ်ကြိမ်ထုတ်ဝေသည်။ အခြားမဂ္ဂဇင်ဖြစ်သော The Juggler သည် တစ်နှစ်လျှင် နှစ်ကြိမ်ထ                                                              ထု                                                                                                       ထုတ်ပြန်ပြီး ကျောင်းသားစာပေနှင့် အနုပညာကကို အာရရုံစစိုက်သည်။ Dome နှစ်စဉ်စာအုပ်ကကို နှစ်စဉ်ထုတ်ဝေခခဲ့သည်။ သတင်းစာများသည် မတူညီသောထုတ်ဝေရေးစိတ်ဝင်စားမှုရှိပြီး The Observer ကကို ထုတ်ဝေပြီး အဓိကအားဖြင့် တက္ကသသိုလ်နှင့်အခြားသတင်းများကကို ဖော်ပြကာ Notre Dame နှင့် Saint Mary's College မှ ကျောင်းသားများမှဖြန့်ဝေသည်။ နောက်ဆဆုံးတွင် Scholastic နှင့် The Observers တတိတို့နှင့်မတူဘဲ The Observert သည် လွတ်လပ်သော သိပ္ပံထုတ်ဝေခြင်းဖြစ်ပြီး ပါမောက္ခများ၏အကြံပေးမှု သသိသို့မဟုတ် အယ်ဒီတာတာတာ ကြီးကြပ်မှုမရှိပါ။ ၁၉၈၇ ခုနှစ်တွင်စတင်၍ The Observe"}
...
...
```

လက်ရှိ ဘာသာပြန်ပြီးသလောက် output ဖိုင်ကို ကိုယ့် local စက်ဆီကို ကော်ပီကူးယူပြီး manual checking လုပ်ကြည့်ခဲ့တော့ အောက်ပါအတိုင်း တွေ့ရတယ်...    

```
C:\Users\801680>scp ye@10.99.5.197:/home/ye/ye/exp/gpt-mt/nllb/squad-my/train_context.my Downloads
ye@10.99.5.197's password:
train_context.my                                                                            100%   43KB 141.2KB/s   00:00

C:\Users\801680>
```

```
Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.	ဗိသုကာအရ ကျောင်းသည် ကက်သလစ်ဘာသာဝင်ဖြစ်ပါသည်။ ဗိမာန်အကြီးဆုံး၏ ရွှေဂူပလာတွင် ဗာနုမာရီဘုရား၏ ရွှေရောင်ရုပ်တုတစ်ခုရှိသည်။ ဗာနာနုမဟာမိတ်တော်၏ ရှေ့တွင် ဗားနုမေရီဘုရားရှိရာ၏ လက်မောင်းများဖြင့် ခရစ်တော်၏ ကြေးနီရုပ်တုတစ်ရုပ်ရှိသည်။
Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.	ဗိသုကာအရ ကျောင်းသည် ကက်သလစ်ဘာသာဝင်ဖြစ်ပါသည်။ ဗိမာန်အကြီးဆုံး၏ ရွှေဂူပလာတွင် ဗာနုမာရီဘုရား၏ ရွှေရောင်ရုပ်တုတစ်ခုရှိသည်။ ဗာနာနုမဟာမိတ်တော်၏ ရှေ့တွင် ဗားနုမေရီဘုရားရှိရာ၏ လက်မောင်းများဖြင့် ခရစ်တော်၏ ကြေးနီရုပ်တုတစ်ရုပ်ရှိသည်။
Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.	ဗိသုကာအရ ကျောင်းသည် ကက်သလစ်ဘာသာဝင်ဖြစ်ပါသည်။ ဗိမာန်အကြီးဆုံး၏ ရွှေဂူပလာတွင် ဗာနုမာရီဘုရား၏ ရွှေရောင်ရုပ်တုတစ်ခုရှိသည်။ ဗာနာနုမဟာမိတ်တော်၏ ရှေ့တွင် ဗားနုမေရီဘုရားရှိရာ၏ လက်မောင်းများဖြင့် ခရစ်တော်၏ ကြေးနီရုပ်တုတစ်ရုပ်ရှိသည်။
Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.	ဗိသုကာအရ ကျောင်းသည် ကက်သလစ်ဘာသာဝင်ဖြစ်ပါသည်။ ဗိမာန်အကြီးဆုံး၏ ရွှေဂူပလာတွင် ဗာနုမာရီဘုရား၏ ရွှေရောင်ရုပ်တုတစ်ခုရှိသည်။ ဗာနာနုမဟာမိတ်တော်၏ ရှေ့တွင် ဗားနုမေရီဘုရားရှိရာ၏ လက်မောင်းများဖြင့် ခရစ်တော်၏ ကြေးနီရုပ်တုတစ်ရုပ်ရှိသည်။
Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.	ဗိသုကာအရ ကျောင်းသည် ကက်သလစ်ဘာသာဝင်ဖြစ်ပါသည်။ ဗိမာန်အကြီးဆုံး၏ ရွှေဂူပလာတွင် ဗာနုမာရီဘုရား၏ ရွှေရောင်ရုပ်တုတစ်ခုရှိသည်။ ဗာနာနုမဟာမိတ်တော်၏ ရှေ့တွင် ဗားနုမေရီဘုရားရှိရာ၏ လက်မောင်းများဖြင့် ခရစ်တော်၏ ကြေးနီရုပ်တုတစ်ရုပ်ရှိသည်။
As at most other universities, Notre Dame's students run a number of news media outlets. The nine student-run outlets include three newspapers, both a radio and television station, and several magazines and journals. Begun as a one-page journal in September 1876, the Scholastic magazine is issued twice monthly and claims to be the oldest continuous collegiate publication in the United States. The other magazine, The Juggler, is released twice a year and focuses on student literature and artwork. The Dome yearbook is published annually. The newspapers have varying publication interests, with The Observer published daily and mainly reporting university and other news, and staffed by students from both Notre Dame and Saint Mary's College. Unlike Scholastic and The Dome, The Observer is an independent publication and does not have a faculty advisor or any editorial oversight from the University. In 1987, when some students believed that The Observer began to show a conservative bias, a liberal newspaper, Common Sense was published. Likewise, in 2003, when other students believed that the paper showed a liberal bias, the conservative paper Irish Rover went into production. Neither paper is published as often as The Observer; however, all three are distributed to all students. Finally, in Spring 2008 an undergraduate journal for political science research, Beyond Politics, made its debut.	Notre Dame ၏ကျောင်းသားများသည် သတင်းမီဒီယာအတော်များများကို လည်ပတ်သည်။ ကျောင်းသားများက လည်ပတ်သော ၉ ခုသော သတင်းဌာနများတွင် ရေဒီယိုနှင့် ရုပ်မြင်သံကြားဌာနတစ်ခုနှင့် မဂ္ဂဇင်းနှင့် ဂျာနယ်များစွာပါဝင်သည်။ ၁၈၇၆ ခုနှစ် စက်တင်ဘာလတွင် စာမျက်နှာတစ်မျက်နှာစာ ဂျာနလအဖြစ်စတင်ထုတ်ဝေခဲ့ပြီး အမေရိကန်ပြည်ထောင်စုတွင် ရှေးကျဆုံးဆက်တိုက် ကောလိပ်ထုတ်ဝေမှုဖြစ်သည်ဟုဆိုကာ လစဉ်နှစ်ကြိမ်ထုတ်ဝေသည်။ အခြားမဂ္ဂဇင်ဖြစ်သော The Juggler သည် တစ်နှစ်လျှင် နှစ်ကြိမ်ထုတ်ပြန်ပြီး ကျောင်းသားစာပေနှင့် အနုပညာကို အာရုံစိုက်သည်။ Dome နှစ်စဉ်စာအုပ်ကို နှစ်စဉ်ထုတ်ဝေခဲ့သည်။ သတင်းစာများသည် မတူညီသောထုတ်ဝေရေးစိတ်ဝင်စားမှုရှိပြီး The Observer ကို ထုတ်ဝေပြီး အဓိကအားဖြင့် တက္ကသိုလ်နှင့်အခြားသတင်းများကို ဖော်ပြကာ Notre Dame နှင့် Saint Mary's College မှ ကျောင်းသားများမှဖြန့်ဝေသည်။ နောက်ဆုံးတွင် Scholastic နှင့် The Observers တို့နှင့်မတူဘဲ The Observert သည် လွတ်လပ်သော သိပ္ပံထုတ်ဝေခြင်းဖြစ်ပြီး ပါမောက္ခများ၏အကြံပေးမှု သို့မဟုတ် အယ်ဒီတာတာတာ ကြီးကြပ်မှုမရှိပါ။ ၁၉၈၇ ခုနှစ်တွင်စတင်၍ The Observe
As at most other universities, Notre Dame's students run a number of news media outlets. The nine student-run outlets include three newspapers, both a radio and television station, and several magazines and journals. Begun as a one-page journal in September 1876, the Scholastic magazine is issued twice monthly and claims to be the oldest continuous collegiate publication in the United States. The other magazine, The Juggler, is released twice a year and focuses on student literature and artwork. The Dome yearbook is published annually. The newspapers have varying publication interests, with The Observer published daily and mainly reporting university and other news, and staffed by students from both Notre Dame and Saint Mary's College. Unlike Scholastic and The Dome, The Observer is an independent publication and does not have a faculty advisor or any editorial oversight from the University. In 1987, when some students believed that The Observer began to show a conservative bias, a liberal newspaper, Common Sense was published. Likewise, in 2003, when other students believed that the paper showed a liberal bias, the conservative paper Irish Rover went into production. Neither paper is published as often as The Observer; however, all three are distributed to all students. Finally, in Spring 2008 an undergraduate journal for political science research, Beyond Politics, made its debut.	Notre Dame ၏ကျောင်းသားများသည် သတင်းမီဒီယာအတော်များများကို လည်ပတ်သည်။ ကျောင်းသားများက လည်ပတ်သော ၉ ခုသော သတင်းဌာနများတွင် ရေဒီယိုနှင့် ရုပ်မြင်သံကြားဌာနတစ်ခုနှင့် မဂ္ဂဇင်းနှင့် ဂျာနယ်များစွာပါဝင်သည်။ ၁၈၇၆ ခုနှစ် စက်တင်ဘာလတွင် စာမျက်နှာတစ်မျက်နှာစာ ဂျာနလအဖြစ်စတင်ထုတ်ဝေခဲ့ပြီး အမေရိကန်ပြည်ထောင်စုတွင် ရှေးကျဆုံးဆက်တိုက် ကောလိပ်ထုတ်ဝေမှုဖြစ်သည်ဟုဆိုကာ လစဉ်နှစ်ကြိမ်ထုတ်ဝေသည်။ အခြားမဂ္ဂဇင်ဖြစ်သော The Juggler သည် တစ်နှစ်လျှင် နှစ်ကြိမ်ထုတ်ပြန်ပြီး ကျောင်းသားစာပေနှင့် အနုပညာကို အာရုံစိုက်သည်။ Dome နှစ်စဉ်စာအုပ်ကို နှစ်စဉ်ထုတ်ဝေခဲ့သည်။ သတင်းစာများသည် မတူညီသောထုတ်ဝေရေးစိတ်ဝင်စားမှုရှိပြီး The Observer ကို ထုတ်ဝေပြီး အဓိကအားဖြင့် တက္ကသိုလ်နှင့်အခြားသတင်းများကို ဖော်ပြကာ Notre Dame နှင့် Saint Mary's College မှ ကျောင်းသားများမှဖြန့်ဝေသည်။ နောက်ဆုံးတွင် Scholastic နှင့် The Observers တို့နှင့်မတူဘဲ The Observert သည် လွတ်လပ်သော သိပ္ပံထုတ်ဝေခြင်းဖြစ်ပြီး ပါမောက္ခများ၏အကြံပေးမှု သို့မဟုတ် အယ်ဒီတာတာတာ ကြီးကြပ်မှုမရှိပါ။ ၁၉၈၇ ခုနှစ်တွင်စတင်၍ The Observe
As at most other universities, Notre Dame's students run a number of news media outlets. The nine student-run outlets include three newspapers, both a radio and television station, and several magazines and journals. Begun as a one-page journal in September 1876, the Scholastic magazine is issued twice monthly and claims to be the oldest continuous collegiate publication in the United States. The other magazine, The Juggler, is released twice a year and focuses on student literature and artwork. The Dome yearbook is published annually. The newspapers have varying publication interests, with The Observer published daily and mainly reporting university and other news, and staffed by students from both Notre Dame and Saint Mary's College. Unlike Scholastic and The Dome, The Observer is an independent publication and does not have a faculty advisor or any editorial oversight from the University. In 1987, when some students believed that The Observer began to show a conservative bias, a liberal newspaper, Common Sense was published. Likewise, in 2003, when other students believed that the paper showed a liberal bias, the conservative paper Irish Rover went into production. Neither paper is published as often as The Observer; however, all three are distributed to all students. Finally, in Spring 2008 an undergraduate journal for political science research, Beyond Politics, made its debut.	Notre Dame ၏ကျောင်းသားများသည် သတင်းမီဒီယာအတော်များများကို လည်ပတ်သည်။ ကျောင်းသားများက လည်ပတ်သော ၉ ခုသော သတင်းဌာနများတွင် ရေဒီယိုနှင့် ရုပ်မြင်သံကြားဌာနတစ်ခုနှင့် မဂ္ဂဇင်းနှင့် ဂျာနယ်များစွာပါဝင်သည်။ ၁၈၇၆ ခုနှစ် စက်တင်ဘာလတွင် စာမျက်နှာတစ်မျက်နှာစာ ဂျာနလအဖြစ်စတင်ထုတ်ဝေခဲ့ပြီး အမေရိကန်ပြည်ထောင်စုတွင် ရှေးကျဆုံးဆက်တိုက် ကောလိပ်ထုတ်ဝေမှုဖြစ်သည်ဟုဆိုကာ လစဉ်နှစ်ကြိမ်ထုတ်ဝေသည်။ အခြားမဂ္ဂဇင်ဖြစ်သော The Juggler သည် တစ်နှစ်လျှင် နှစ်ကြိမ်ထုတ်ပြန်ပြီး ကျောင်းသားစာပေနှင့် အနုပညာကို အာရုံစိုက်သည်။ Dome နှစ်စဉ်စာအုပ်ကို နှစ်စဉ်ထုတ်ဝေခဲ့သည်။ သတင်းစာများသည် မတူညီသောထုတ်ဝေရေးစိတ်ဝင်စားမှုရှိပြီး The Observer ကို ထုတ်ဝေပြီး အဓိကအားဖြင့် တက္ကသိုလ်နှင့်အခြားသတင်းများကို ဖော်ပြကာ Notre Dame နှင့် Saint Mary's College မှ ကျောင်းသားများမှဖြန့်ဝေသည်။ နောက်ဆုံးတွင် Scholastic နှင့် The Observers တို့နှင့်မတူဘဲ The Observert သည် လွတ်လပ်သော သိပ္ပံထုတ်ဝေခြင်းဖြစ်ပြီး ပါမောက္ခများ၏အကြံပေးမှု သို့မဟုတ် အယ်ဒီတာတာတာ ကြီးကြပ်မှုမရှိပါ။ ၁၉၈၇ ခုနှစ်တွင်စတင်၍ The Observe
As at most other universities, Notre Dame's students run a number of news media outlets. The nine student-run outlets include three newspapers, both a radio and television station, and several magazines and journals. Begun as a one-page journal in September 1876, the Scholastic magazine is issued twice monthly and claims to be the oldest continuous collegiate publication in the United States. The other magazine, The Juggler, is released twice a year and focuses on student literature and artwork. The Dome yearbook is published annually. The newspapers have varying publication interests, with The Observer published daily and mainly reporting university and other news, and staffed by students from both Notre Dame and Saint Mary's College. Unlike Scholastic and The Dome, The Observer is an independent publication and does not have a faculty advisor or any editorial oversight from the University. In 1987, when some students believed that The Observer began to show a conservative bias, a liberal newspaper, Common Sense was published. Likewise, in 2003, when other students believed that the paper showed a liberal bias, the conservative paper Irish Rover went into production. Neither paper is published as often as The Observer; however, all three are distributed to all students. Finally, in Spring 2008 an undergraduate journal for political science research, Beyond Politics, made its debut.	Notre Dame ၏ကျောင်းသားများသည် သတင်းမီဒီယာအတော်များများကို လည်ပတ်သည်။ ကျောင်းသားများက လည်ပတ်သော ၉ ခုသော သတင်းဌာနများတွင် ရေဒီယိုနှင့် ရုပ်မြင်သံကြားဌာနတစ်ခုနှင့် မဂ္ဂဇင်းနှင့် ဂျာနယ်များစွာပါဝင်သည်။ ၁၈၇၆ ခုနှစ် စက်တင်ဘာလတွင် စာမျက်နှာတစ်မျက်နှာစာ ဂျာနလအဖြစ်စတင်ထုတ်ဝေခဲ့ပြီး အမေရိကန်ပြည်ထောင်စုတွင် ရှေးကျဆုံးဆက်တိုက် ကောလိပ်ထုတ်ဝေမှုဖြစ်သည်ဟုဆိုကာ လစဉ်နှစ်ကြိမ်ထုတ်ဝေသည်။ အခြားမဂ္ဂဇင်ဖြစ်သော The Juggler သည် တစ်နှစ်လျှင် နှစ်ကြိမ်ထုတ်ပြန်ပြီး ကျောင်းသားစာပေနှင့် အနုပညာကို အာရုံစိုက်သည်။ Dome နှစ်စဉ်စာအုပ်ကို နှစ်စဉ်ထုတ်ဝေခဲ့သည်။ သတင်းစာများသည် မတူညီသောထုတ်ဝေရေးစိတ်ဝင်စားမှုရှိပြီး The Observer ကို ထုတ်ဝေပြီး အဓိကအားဖြင့် တက္ကသိုလ်နှင့်အခြားသတင်းများကို ဖော်ပြကာ Notre Dame နှင့် Saint Mary's College မှ ကျောင်းသားများမှဖြန့်ဝေသည်။ နောက်ဆုံးတွင် Scholastic နှင့် The Observers တို့နှင့်မတူဘဲ The Observert သည် လွတ်လပ်သော သိပ္ပံထုတ်ဝေခြင်းဖြစ်ပြီး ပါမောက္ခများ၏အကြံပေးမှု သို့မဟုတ် အယ်ဒီတာတာတာ ကြီးကြပ်မှုမရှိပါ။ ၁၉၈၇ ခုနှစ်တွင်စတင်၍ The Observe
As at most other universities, Notre Dame's students run a number of news media outlets. The nine student-run outlets include three newspapers, both a radio and television station, and several magazines and journals. Begun as a one-page journal in September 1876, the Scholastic magazine is issued twice monthly and claims to be the oldest continuous collegiate publication in the United States. The other magazine, The Juggler, is released twice a year and focuses on student literature and artwork. The Dome yearbook is published annually. The newspapers have varying publication interests, with The Observer published daily and mainly reporting university and other news, and staffed by students from both Notre Dame and Saint Mary's College. Unlike Scholastic and The Dome, The Observer is an independent publication and does not have a faculty advisor or any editorial oversight from the University. In 1987, when some students believed that The Observer began to show a conservative bias, a liberal newspaper, Common Sense was published. Likewise, in 2003, when other students believed that the paper showed a liberal bias, the conservative paper Irish Rover went into production. Neither paper is published as often as The Observer; however, all three are distributed to all students. Finally, in Spring 2008 an undergraduate journal for political science research, Beyond Politics, made its debut.	Notre Dame ၏ကျောင်းသားများသည် သတင်းမီဒီယာအတော်များများကို လည်ပတ်သည်။ ကျောင်းသားများက လည်ပတ်သော ၉ ခုသော သတင်းဌာနများတွင် ရေဒီယိုနှင့် ရုပ်မြင်သံကြားဌာနတစ်ခုနှင့် မဂ္ဂဇင်းနှင့် ဂျာနယ်များစွာပါဝင်သည်။ ၁၈၇၆ ခုနှစ် စက်တင်ဘာလတွင် စာမျက်နှာတစ်မျက်နှာစာ ဂျာနလအဖြစ်စတင်ထုတ်ဝေခဲ့ပြီး အမေရိကန်ပြည်ထောင်စုတွင် ရှေးကျဆုံးဆက်တိုက် ကောလိပ်ထုတ်ဝေမှုဖြစ်သည်ဟုဆိုကာ လစဉ်နှစ်ကြိမ်ထုတ်ဝေသည်။ အခြားမဂ္ဂဇင်ဖြစ်သော The Juggler သည် တစ်နှစ်လျှင် နှစ်ကြိမ်ထုတ်ပြန်ပြီး ကျောင်းသားစာပေနှင့် အနုပညာကို အာရုံစိုက်သည်။ Dome နှစ်စဉ်စာအုပ်ကို နှစ်စဉ်ထုတ်ဝေခဲ့သည်။ သတင်းစာများသည် မတူညီသောထုတ်ဝေရေးစိတ်ဝင်စားမှုရှိပြီး The Observer ကို ထုတ်ဝေပြီး အဓိကအားဖြင့် တက္ကသိုလ်နှင့်အခြားသတင်းများကို ဖော်ပြကာ Notre Dame နှင့် Saint Mary's College မှ ကျောင်းသားများမှဖြန့်ဝေသည်။ နောက်ဆုံးတွင် Scholastic နှင့် The Observers တို့နှင့်မတူဘဲ The Observert သည် လွတ်လပ်သော သိပ္ပံထုတ်ဝေခြင်းဖြစ်ပြီး ပါမောက္ခများ၏အကြံပေးမှု သို့မဟုတ် အယ်ဒီတာတာတာ ကြီးကြပ်မှုမရှိပါ။ ၁၉၈၇ ခုနှစ်တွင်စတင်၍ The Observe
The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome). Its main seminary, Moreau Seminary, is located on the campus across St. Joseph lake from the Main Building. Old College, the oldest building on campus and located near the shore of St. Mary lake, houses undergraduate seminarians. Retired priests and brothers reside in Fatima House (a former retreat center), Holy Cross House, as well as Columba Hall near the Grotto. The university through the Moreau Seminary has ties to theologian Frederick Buechner. While not Catholic, Buechner has praised writers from Notre Dame and Moreau Seminary created a Buechner Prize for Preaching.	ရိုမန်မြို့တွင် တည်ရှိသည်။ ၎င်း၏အဓိကသင်ကျောင်း Moreau Seminary သည်ဗိမာန်တွင်ရှိပြီး ဗိမာန္၏အရင်ဆုံးအဆောက်အအုံဖြစ်သော Old College သည်ဗီမာန်တော်၏အကြီးဆုံးအဆင့်ဖြစ်ကာ St. Mary ကမ်းခြေအနီးတွင် တည်ရှိနေသည်။ အငြိမ်းစားရဟန်းများနှင့်ညီအစ်ကိုများသည် Fatima House (ယခင်က ပြန်လည်ဆုတ်ခွာရေးစင်တာတစ်ခု) ၊ Holy Cross House နှင့် Grotto အနီးရှိ Columba Hall တွင်နေထိုင်သည်။ Moreau သင်ကျောင်းမှတဆင့်တက္ကသိုလ်သည် ဘာသာဗေဒပညာရှင် Frederick Buechner နှင့်ဆက်နွယ်မှုရှိသည်။ ကက်သလစ်မဟုတ်သော်လည်း Buechtner သည် Notre Dame မှစာရေးဆရာများကို ချီးကျူးခဲ့ပြီး Moreau ဘာသာရေးကျောင်းသည်ဟောပြောခြင်းအတွက် Buechenner ဆုတစ်ခုဖန်တီးခဲ့သည်။
The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome). Its main seminary, Moreau Seminary, is located on the campus across St. Joseph lake from the Main Building. Old College, the oldest building on campus and located near the shore of St. Mary lake, houses undergraduate seminarians. Retired priests and brothers reside in Fatima House (a former retreat center), Holy Cross House, as well as Columba Hall near the Grotto. The university through the Moreau Seminary has ties to theologian Frederick Buechner. While not Catholic, Buechner has praised writers from Notre Dame and Moreau Seminary created a Buechner Prize for Preaching.	ရိုမန်မြို့တွင် တည်ရှိသည်။ ၎င်း၏အဓိကသင်ကျောင်း Moreau Seminary သည်ဗိမာန်တွင်ရှိပြီး ဗိမာန္၏အရင်ဆုံးအဆောက်အအုံဖြစ်သော Old College သည်ဗီမာန်တော်၏အကြီးဆုံးအဆင့်ဖြစ်ကာ St. Mary ကမ်းခြေအနီးတွင် တည်ရှိနေသည်။ အငြိမ်းစားရဟန်းများနှင့်ညီအစ်ကိုများသည် Fatima House (ယခင်က ပြန်လည်ဆုတ်ခွာရေးစင်တာတစ်ခု) ၊ Holy Cross House နှင့် Grotto အနီးရှိ Columba Hall တွင်နေထိုင်သည်။ Moreau သင်ကျောင်းမှတဆင့်တက္ကသိုလ်သည် ဘာသာဗေဒပညာရှင် Frederick Buechner နှင့်ဆက်နွယ်မှုရှိသည်။ ကက်သလစ်မဟုတ်သော်လည်း Buechtner သည် Notre Dame မှစာရေးဆရာများကို ချီးကျူးခဲ့ပြီး Moreau ဘာသာရေးကျောင်းသည်ဟောပြောခြင်းအတွက် Buechenner ဆုတစ်ခုဖန်တီးခဲ့သည်။
The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome). Its main seminary, Moreau Seminary, is located on the campus across St. Joseph lake from the Main Building. Old College, the oldest building on campus and located near the shore of St. Mary lake, houses undergraduate seminarians. Retired priests and brothers reside in Fatima House (a former retreat center), Holy Cross House, as well as Columba Hall near the Grotto. The university through the Moreau Seminary has ties to theologian Frederick Buechner. While not Catholic, Buechner has praised writers from Notre Dame and Moreau Seminary created a Buechner Prize for Preaching.	ရိုမန်မြို့တွင် တည်ရှိသည်။ ၎င်း၏အဓိကသင်ကျောင်း Moreau Seminary သည်ဗိမာန်တွင်ရှိပြီး ဗိမာန္၏အရင်ဆုံးအဆောက်အအုံဖြစ်သော Old College သည်ဗီမာန်တော်၏အကြီးဆုံးအဆင့်ဖြစ်ကာ St. Mary ကမ်းခြေအနီးတွင် တည်ရှိနေသည်။ အငြိမ်းစားရဟန်းများနှင့်ညီအစ်ကိုများသည် Fatima House (ယခင်က ပြန်လည်ဆုတ်ခွာရေးစင်တာတစ်ခု) ၊ Holy Cross House နှင့် Grotto အနီးရှိ Columba Hall တွင်နေထိုင်သည်။ Moreau သင်ကျောင်းမှတဆင့်တက္ကသိုလ်သည် ဘာသာဗေဒပညာရှင် Frederick Buechner နှင့်ဆက်နွယ်မှုရှိသည်။ ကက်သလစ်မဟုတ်သော်လည်း Buechtner သည် Notre Dame မှစာရေးဆရာများကို ချီးကျူးခဲ့ပြီး Moreau ဘာသာရေးကျောင်းသည်ဟောပြောခြင်းအတွက် Buechenner ဆုတစ်ခုဖန်တီးခဲ့သည်။
The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome). Its main seminary, Moreau Seminary, is located on the campus across St. Joseph lake from the Main Building. Old College, the oldest building on campus and located near the shore of St. Mary lake, houses undergraduate seminarians. Retired priests and brothers reside in Fatima House (a former retreat center), Holy Cross House, as well as Columba Hall near the Grotto. The university through the Moreau Seminary has ties to theologian Frederick Buechner. While not Catholic, Buechner has praised writers from Notre Dame and Moreau Seminary created a Buechner Prize for Preaching.	ရိုမန်မြို့တွင် တည်ရှိသည်။ ၎င်း၏အဓိကသင်ကျောင်း Moreau Seminary သည်ဗိမာန်တွင်ရှိပြီး ဗိမာန္၏အရင်ဆုံးအဆောက်အအုံဖြစ်သော Old College သည်ဗီမာန်တော်၏အကြီးဆုံးအဆင့်ဖြစ်ကာ St. Mary ကမ်းခြေအနီးတွင် တည်ရှိနေသည်။ အငြိမ်းစားရဟန်းများနှင့်ညီအစ်ကိုများသည် Fatima House (ယခင်က ပြန်လည်ဆုတ်ခွာရေးစင်တာတစ်ခု) ၊ Holy Cross House နှင့် Grotto အနီးရှိ Columba Hall တွင်နေထိုင်သည်။ Moreau သင်ကျောင်းမှတဆင့်တက္ကသိုလ်သည် ဘာသာဗေဒပညာရှင် Frederick Buechner နှင့်ဆက်နွယ်မှုရှိသည်။ ကက်သလစ်မဟုတ်သော်လည်း Buechtner သည် Notre Dame မှစာရေးဆရာများကို ချီးကျူးခဲ့ပြီး Moreau ဘာသာရေးကျောင်းသည်ဟောပြောခြင်းအတွက် Buechenner ဆုတစ်ခုဖန်တီးခဲ့သည်။
The university is the major seat of the Congregation of Holy Cross (albeit not its official headquarters, which are in Rome). Its main seminary, Moreau Seminary, is located on the campus across St. Joseph lake from the Main Building. Old College, the oldest building on campus and located near the shore of St. Mary lake, houses undergraduate seminarians. Retired priests and brothers reside in Fatima House (a former retreat center), Holy Cross House, as well as Columba Hall near the Grotto. The university through the Moreau Seminary has ties to theologian Frederick Buechner. While not Catholic, Buechner has praised writers from Notre Dame and Moreau Seminary created a Buechner Prize for Preaching.	ရိုမန်မြို့တွင် တည်ရှိသည်။ ၎င်း၏အဓိကသင်ကျောင်း Moreau Seminary သည်ဗိမာန်တွင်ရှိပြီး ဗိမာန္၏အရင်ဆုံးအဆောက်အအုံဖြစ်သော Old College သည်ဗီမာန်တော်၏အကြီးဆုံးအဆင့်ဖြစ်ကာ St. Mary ကမ်းခြေအနီးတွင် တည်ရှိနေသည်။ အငြိမ်းစားရဟန်းများနှင့်ညီအစ်ကိုများသည် Fatima House (ယခင်က ပြန်လည်ဆုတ်ခွာရေးစင်တာတစ်ခု) ၊ Holy Cross House နှင့် Grotto အနီးရှိ Columba Hall တွင်နေထိုင်သည်။ Moreau သင်ကျောင်းမှတဆင့်တက္ကသိုလ်သည် ဘာသာဗေဒပညာရှင် Frederick Buechner နှင့်ဆက်နွယ်မှုရှိသည်။ ကက်သလစ်မဟုတ်သော်လည်း Buechtner သည် Notre Dame မှစာရေးဆရာများကို ချီးကျူးခဲ့ပြီး Moreau ဘာသာရေးကျောင်းသည်ဟောပြောခြင်းအတွက် Buechenner ဆုတစ်ခုဖန်တီးခဲ့သည်။
The College of Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study – aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering – with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively.	စက်မှုတက္ကသိုလ်သည် ၁၉၂၀ ခုနှစ်တွင် တည်ထောင်ခဲ့သော်လည်း ၁၉၇၀ ခုနှစ်မှစ၍ အရပ်ဘက်နှင့် စက်မှုအင်ဂျင်နီယာပညာရပ်များတွင် အစောပိုင်းသင်တန်းများသည် သိပ္ပံတက္ကစီ၏ အစိတ်အပိုင်းတစ်ခုဖြစ်သည်။ ယနေ့တွင် Fitzpatrick, Cushing နှင့် Stinson-Remick Halls of Engineering တွင် တည်ရှိသည့် ကောလိပ်တွင် ပညာရေးဌာနငါးခုပါဝင်သည်။ လေကြောင်းနှင့် စက်ပစ္စည်းအင်ဂျင်နီးယာ၊ ဓာတုနှင့် ဇီဝမော်လီကျူးအင်ဂျင်ဂျင်နီယား၊ အရပ်ဖက်နှင့် ဘူမိဗေဒသိပ္ပံ၊ ကွန်ပျူတာသိပ္ပံနှင့် အင်ဂျင်နီယရီနှင့် လျှပ်စစ်အင်ဂျင်စီယာ ၈ ခုဖြင့် BS ဘွဲ့များပေးသည်။ ထို့ပြင် ကောလိယပ်သည်အခြားအထူးတန်းများဖြစ်သော B.A နှင့် Master of Business Administration (MBA) ဘွဲ့များကိုပေးသည်။
The College of Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study – aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering – with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively.	စက်မှုတက္ကသိုလ်သည် ၁၉၂၀ ခုနှစ်တွင် တည်ထောင်ခဲ့သော်လည်း ၁၉၇၀ ခုနှစ်မှစ၍ အရပ်ဘက်နှင့် စက်မှုအင်ဂျင်နီယာပညာရပ်များတွင် အစောပိုင်းသင်တန်းများသည် သိပ္ပံတက္ကစီ၏ အစိတ်အပိုင်းတစ်ခုဖြစ်သည်။ ယနေ့တွင် Fitzpatrick, Cushing နှင့် Stinson-Remick Halls of Engineering တွင် တည်ရှိသည့် ကောလိပ်တွင် ပညာရေးဌာနငါးခုပါဝင်သည်။ လေကြောင်းနှင့် စက်ပစ္စည်းအင်ဂျင်နီးယာ၊ ဓာတုနှင့် ဇီဝမော်လီကျူးအင်ဂျင်ဂျင်နီယား၊ အရပ်ဖက်နှင့် ဘူမိဗေဒသိပ္ပံ၊ ကွန်ပျူတာသိပ္ပံနှင့် အင်ဂျင်နီယရီနှင့် လျှပ်စစ်အင်ဂျင်စီယာ ၈ ခုဖြင့် BS ဘွဲ့များပေးသည်။ ထို့ပြင် ကောလိယပ်သည်အခြားအထူးတန်းများဖြစ်သော B.A နှင့် Master of Business Administration (MBA) ဘွဲ့များကိုပေးသည်။
The College of Engineering was established in 1920, however, early courses in civil and mechanical engineering were a part of the College of Science since the 1870s. Today the college, housed in the Fitzpatrick, Cushing, and Stinson-Remick Halls of Engineering, includes five departments of study – aerospace and mechanical engineering, chemical and biomolecular engineering, civil engineering and geological sciences, computer science and engineering, and electrical engineering – with eight B.S. degrees offered. Additionally, the college offers five-year dual degree programs with the Colleges of Arts and Letters and of Business awarding additional B.A. and Master of Business Administration (MBA) degrees, respectively.	စက်မှုတက္ကသိုလ်သည် ၁၉၂၀ ခုနှစ်တွင် တည်ထောင်ခဲ့သော်လည်း ၁၉၇၀ ခုနှစ်မှစ၍ အရပ်ဘက်နှင့် စက်မှုအင်ဂျင်နီယာပညာရပ်များတွင် အစောပိုင်းသင်တန်းများသည် သိပ္ပံတက္ကစီ၏ အစိတ်အပိုင်းတစ်ခုဖြစ်သည်။ ယနေ့တွင် Fitzpatrick, Cushing နှင့် Stinson-Remick Halls of Engineering တွင် တည်ရှိသည့် ကောလိပ်တွင် ပညာရေးဌာနငါးခုပါဝင်သည်။ လေကြောင်းနှင့် စက်ပစ္စည်းအင်ဂျင်နီးယာ၊ ဓာတုနှင့် ဇီဝမော်လီကျူးအင်ဂျင်ဂျင်နီယား၊ အရပ်ဖက်နှင့် ဘူမိဗေဒသိပ္ပံ၊ ကွန်ပျူတာသိပ္ပံနှင့် အင်ဂျင်နီယရီနှင့် လျှပ်စစ်အင်ဂျင်စီယာ ၈ ခုဖြင့် BS ဘွဲ့များပေးသည်။ ထို့ပြင် ကောလိယပ်သည်အခြားအထူးတန်းများဖြစ်သော B.A နှင့် Master of Business Administration (MBA) ဘွဲ့များကိုပေးသည်။

```

အထက်မှာ မြင်ရတဲ့အတိုင်းပဲ မြန်မာလိုဘာသာပြန်ပြီး ထွက်လာတဲ့ output က အရမ်းရှည်တဲ့စာကြောင်းတွေမှာဆိုရင် ဘာသာပြန်အမှားတွေ ပိုများတာကို တွေ့ရတယ်။  

## Shell Script Preparation for SQUAD Validation Dataset  

validation dataset ကို ဘာသာပြန်ဖို့ လိုအပ်တာမို့ shell script ပြင်ဆင်ခဲ့...  

```bash
#!/bin/bash

# Base directory for input files
INPUT_DIR="/home/ye/ye/exp/gpt-mt/nllb/data/squad/valid/"

# Directory for output files
OUTPUT_DIR="/home/ye/ye/exp/gpt-mt/nllb/squad-my/"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Iterate over each .src file in the input directory
for FILE in "$INPUT_DIR"/*.txt; do
    # Extract the base filename without the extension
    BASENAME=$(basename "$FILE" .txt)

    # Define the output file name
    OUTPUT_FILE="$OUTPUT_DIR/$BASENAME.my"

    # Print the command being executed (for debugging)
    echo "Running nllb-translate.sh for $FILE"

    # Run the translation command
    time ./nllb-translate.sh --input "$FILE" --source eng_Latn --target mya_Mymr --output "$OUTPUT_FILE"
done
```

nllb-translate.sh ဆိုတဲ့ ဖိုင်ကတော့ ကိုယ့်ဖိုလ်ဒါထဲမှာ ပြင်ဆင်ထားဖို့ လိုအပ်တယ်။  

## NLLB Translation for SQUAD Validation Dataset  

screen command ကြို run ထားဖို့ မမေ့နဲ့...  

```
screen -S squad_valid
```

လက်ရှိ run ထားတဲ့ scren တွေကို ကြည့်ချင်ရင်...  

```
(base) ye@lst-gpu-server-197:~$ screen -ls
There are screens on:
        1955959.squad_valid     (06/12/2025 03:04:55 PM)        (Attached)
        1954893.squad   (06/12/2025 02:41:48 PM)        (Attached)
2 Sockets in /run/screen/S-ye.
(base) ye@lst-gpu-server-197:~$
```

Attached ဖြစ်နေရင် detach လုပ်ပြီးမှ ပြန်ချိတ်ပါ။  

```
(base) ye@lst-gpu-server-197:~$ screen -D squad_valid
[1955959.squad_valid power detached.]
```

ပြန်ချိတ်မယ် ဆိုရင်တော့...  

```
(base) ye@lst-gpu-server-197:~$ screen -R squad_valid
```

Start translation for validation dataset...  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb$ ./squad_valid2my.sh | tee squad_valid_tran.log
Running nllb-translate.sh for /home/ye/ye/exp/gpt-mt/nllb/data/squad/valid//valid_context.txt
JSON Payload: {
  "text": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"Super Bowl 50 သည် ၂၀၁၅ ရာသီအတွက် အမျျိုးသားဘောလလုံးလိဂ် (NFL) ၏ချန်ပီယံကကိုသတ်မှတ်သည့်အမေရိကန်ဘောလလုံးပွဲဖြစ်သည်။ အမေရိကန်ဘောလလုံးကွန်ဖရင့် (AFC) ချန်ပီယယို Denver Broncos သည် အမျျိုးသားဘောဘောလလုံး ကွန်ဖရန်ရှင်း (NFC) ၏ ချန်ဖယို Carolina Panthers ကကို ၂၄၁၀ ဖြင့်အနနိုင်ရပြီး တတိယ Super Bowl ဆုကကိုရရှိခခဲ့သည်။ ဒီပွဲကကို ၂၀၁၆ ခုနှစ် ဖေဖော်ဝါရီလ ၇ ရက်တွင် ကယ်လီဖဖိုးနီးယားပြည်နယ်၊ ဆန်တာကလာရာရှိ ဆန်ဖရန်စစ္စကကိုပင်လယ်ကွေ့ဒေသရှိ လီဗီအားကစားရရုံတွင်ကစားခခဲ့သည်။ ဒါဟာ ၅၀ ကြိမ်မြောက် Super Bowl ဖြစ်သောကြောင့်လိဂ္သည် \"ရွှေနှစ်ပတ်လည်\" ကကိုအလေးထားကာရွှေအကြောင်းအရာများနှင့်အတူအစီအစဉ်အမျျိုးမျျိုးဖြင့်အထူးပြုခခဲ့သည်။ ထထိထို့ပြင် Super Bowl ဂိမ်းတတိုင်းကကို ရောမကိန်းများဖြင့်အမည်ပေးခြင်း၏ရရိုးရာကကို ယာယီရပ်ဆဆိုင်းခခဲ့သည်။"}
JSON Payload: {
  "text": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
...
...
...
JSON Payload: {
  "text": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.",
  "source": "eng_Latn",
  "target": "mya_Mymr"
}
API Response: {"result":"Super Bowl 50 သည် ၂၀၁၅ ရာသီအတွက် အမျျိုးသားဘောလလုံးလိဂ် (NFL) ၏ချန်ပီယံကကိုသတ်မှတ်သည့်အမေရိကန်ဘောလလုံးပွဲဖြစ်သည်။ အမေရိကန်ဘောလလုံးကွန်ဖရင့် (AFC) ချန်ပီယယို Denver Broncos သည် အမျျိုးသားဘောဘောလလုံး ကွန်ဖရန်ရှင်း (NFC) ၏ ချန်ဖယို Carolina Panthers ကကို ၂၄၁၀ ဖြင့်အနနိုင်ရပြီး တတိယ Super Bowl ဆုကကိုရရှိခခဲ့သည်။ ဒီပွဲကကို ၂၀၁၆ ခုနှစ် ဖေဖော်ဝါရီလ ၇ ရက်တွင် ကယ်လီဖဖိုးနီးယားပြည်နယ်၊ ဆန်တာကလာရာရှိ ဆန်ဖရန်စစ္စကကိုပင်လယ်ကွေ့ဒေသရှိ လီဗီအားကစားရရုံတွင်ကစားခခဲ့သည်။ ဒါဟာ ၅၀ ကြိမ်မြောက် Super Bowl ဖြစ်သောကြောင့်လိဂ္သည် \"ရွှေနှစ်ပတ်လည်\" ကကိုအလေးထားကာရွှေအကြောင်းအရာများနှင့်အတူအစီအစဉ်အမျျိုးမျျိုးဖြင့်အထူးပြုခခဲ့သည်။ ထထိထို့ပြင် Super Bowl ဂိမ်းတတိုင်းကကို ရောမကိန်းများဖြင့်အမည်ပေးခြင်း၏ရရိုးရာကကို ယာယီရပ်ဆဆိုင်းခခဲ့သည်။"}
```

screen command ကို run ထားတဲ့ terminal ကနေ မြန်မာစာကြောင်းတွေကို ကော်ပီကူးရင် ထပ်ပါလာတာတွေ၊ မှားနေတာတွေ ဖြစ်တတ်တယ်။ အဲဒါကြောင့် ကြည့်ချင်တဲ့ မြန်မာစာဖိုင်ကို ကိုယ့် local စက်ထဲကို ကော်ပီကူးပြီး ကြည့်မှ မှန်မှန်ကန်ကန် မြင်ရပါလိမ့်မယ်။ သတိထားပါ။  

Run ထားလိုက်ပြီ။ အကြမ်းပြန်ထားတဲ့ output တော့ ရလာနိုင်တယ်။  
တခု ရှိတာက တခါတလေ ဘာသာပြန်မပေးပဲ ကျော်သွားတာမျိုးတွေလည်း ရှိတော့ အဲဒီ အပိုင်းကိုလည်း သတိထား ကြည့်ရလိမ့်မယ်။  
ဘယ်စာကြောင်းတွေကို ဘာသာမပြန်ပဲ သွားသလဲ ဆိုတာကတော့ shell နဲ့ ရေးပြီး check လုပ်နိုင်ဖို့ ပြင်ဆင်ထားပါတယ်။  

```
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/squad-my$ ls
train_context.my  valid_context.my
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/squad-my$
(base) ye@lst-gpu-server-197:~/ye/exp/gpt-mt/nllb/squad-my$ wc *
    84  16918 241737 train_context.my
    25   4425  63975 valid_context.my
```
