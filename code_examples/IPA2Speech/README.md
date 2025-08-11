## Demo Code for IPA to Speech

- ၁၀ ဩဂုတ် ၂၀၂၅ မှာလုပ်ခဲ့တဲ့ seminar မှာ IPA code တွေရဲ့ အသုံးဝင်ပုံ အနေနဲ့ ဟိုးအရင်က လုပ်ခဲ့ကြတဲ့ rule-based TTS (စက်ရုပ်အသံ) ကို ဒီမိုအနေနဲ့ လုပ်ပြဖြစ်ခဲ့တယ်။
- Linux မှာ အသုံးများတဲ့ espeak command (တကယ်က espeak-NG ဗားရှင်းပါ) ဆီကို IPA code ကို စာကြောင်းတစ်ကြောင်းချင်းစီ pass လုပ်တဲ့ ပုံစံအနေနဲ့ လုပ်ပြခဲ့တာပါ
- IPA က myG2P မှာ သုံးထားတဲ့ မြန်မာစာအတွက် ထပ်ဖြည့်တာ လုပ်ထားတဲ့ IPA ပါ
- Espeak မှာက သူဖာသာသူ သတ်မှတ်ထားတဲ့ code က ရှိတာကြောင့် အဲဒီအတွက် mapping ကို ပြင်ရပါတယ် (i.e. myG2P IPA to espeak code)
- ပြီးရင်တော့ espeak ကို pass လုပ်ပြီး TTS လုပ်လို့ ရပါတယ်
- လက်ရှိ အလွယ် ဒီမို လုပ်ပြဖြစ်တာမို့လို့ အသံက robot အသံမှာမှ ဆိုးပါသေးတယ်။ နားလည်ရ ခက်ပါလိမ့်မယ်။ Proof of concept အနေနဲ့ လုပ်ပြခဲ့တာပါ
- ပိုကောင်းချင်ရင်တော့ Espeak မှာ မြန်မာစာအတွက် သပ်သပ် rule တွေ ပြင်ရပါလိမ့်မယ်။
- လက်ရှိမှာ Espeak အတွက် မြန်မာစာကို ပြင်ထားတာ ရှိပေမယ့် မပြီးသေးတာ၊ သို့မဟုတ် မပြည့်စုံသေးတာလို့ နားလည်တယ်။ အဲဒါကြောင့် myG2P IPA ကို English espeak code အနေနဲ့ပဲ ပြောင်းပြီး ဒီမိုလုပ်ပြထားတာပါ

## Espeak Version Information

```
ye@lst-hpc3090:~/exp/vs/ipa2speech$ espeak --version
eSpeak NG text-to-speech: 1.52.0  Data at: /usr/local/share/espeak-ng-data
ye@lst-hpc3090:~/exp/vs/ipa2speech$
```

## --help 

```
ye@lst-hpc3090:~/exp/vs/ipa2speech$ python ./ipa_my2speech.py --help
usage: ipa_my2speech.py [-h] [--input INPUT] [--output OUTPUT] [--voice VOICE] [--speed SPEED]

Convert IPA (International Phonetic Alphabet) to speech using espeak

options:
  -h, --help       show this help message and exit
  --input INPUT    Input file containing IPA text (one per line) (default: None)
  --output OUTPUT  Output WAV file (if not specified, returns audio data) (default: None)
  --voice VOICE    Voice to use for speech synthesis (default: en-us)
  --speed SPEED    Speech speed in words per minute (default: 120)
ye@lst-hpc3090:~/exp/vs/ipa2speech$
```

## IPA Example File  

```
ye@lst-hpc3090:~/exp/vs/ipa2speech$ cat test1.ipa.txt
kə tɕʰɪ̀ɴ
kə já
kə jɪ̀ɴ
tɕʰɪ́ɴ
mʊ̀ɴ
bə mà
jə kʰàɪɴ
ʃáɴ
ye@lst-hpc3090:~/exp/vs/ipa2speech$
```

## IPA2Speech Demo

```
ye@lst-hpc3090:~/exp/vs/ipa2speech$ time python ./ipa_my2speech.py --input ./test1.ipa.txt --output ./test1.ipa.wav
Using espeak version: eSpeak NG text-to-speech: 1.52.0  Data at: /usr/local/share/espeak-ng-data
Processing IPA text: kə tɕʰɪ̀ɴ kə já kə jɪ̀ɴ tɕʰɪ́ɴ mʊ̀ɴ bə mà jə kʰàɪɴ ʃáɴ
Converted IPA to eSpeak: kə tɕʰɪ̀ɴ kə já kə jɪ̀ɴ tɕʰɪ́ɴ mʊ̀ɴ bə mà jə kʰàɪɴ ʃáɴ -> k@ ts_hI_Ln k@ ja_H k@ jI_Ln ts_hI_Hn mU_Ln b@ ma_L j@ k_ha_LIn Sa_Hn
Successfully generated speech and saved to ./test1.ipa.wav

real    0m0.095s
user    0m1.484s
sys     0m0.016s
ye@lst-hpc3090:~/exp/vs/ipa2speech$
```

ထွက်လာတဲ့ wavefile information:  

```
ye@lst-hpc3090:~/exp/vs/ipa2speech$ soxi ./test1.ipa.wav

Input File     : './test1.ipa.wav'
Channels       : 1
Sample Rate    : 22050
Precision      : 16-bit
Duration       : 00:00:04.91 = 108191 samples ~ 367.997 CDDA sectors
File Size      : 216k
Bit Rate       : 353k
Sample Encoding: 16-bit Signed Integer PCM

ye@lst-hpc3090:~/exp/vs/ipa2speech$
```

## Listen and Guess

- IPA ကနေ Speech (i.e. TTS) အဖြစ်ပြောင်းပြီး ထွက်လာတဲ့ wavefile ကို နားထောင်ကြည့်ပြီး ဘာတွေ ပြောနေတာလဲ ဆိုတာကို ခန့်မှန်းကြည့်ပါ။  
- Seminar မှာတော့ ကျောင်းသား တချို့က ခန့်မှန်းနိုင်ခဲ့ပြီး တချို့လည်း နားမလည်ကြပါဘူး :)
