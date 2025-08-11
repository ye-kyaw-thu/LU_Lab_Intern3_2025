# FYI

## Assignment-1

- ကိုယ့်မြန်မာ ကြိယာအသံတွေကို Praat နဲ့ အသံဖမ်းပြီး Vowel Space (သို့) Vowel Chart ဆွဲကြည့်ပါ။
- အသံဖမ်းထားတဲ့ Wave file ရော Praat နဲ့ အလုပ်လုပ်ထားတဲ့ ဖိုင်တွေရော၊ ပြီးတော့ Python or R code နဲ့ ဆွဲပြီး ထွက်လာတဲ့ vowel space ဖိုင်အားလုံးကို folder တစ်ခုအောက်မှာ သိမ်းပြီး zip လုပ်ပြီး ပို့ပေးပါ။
- ဥပမာ အောက်ပါဖိုင်တွေ ပါကိုပါရမယ်
  - ye_vowels.wav
  - ye_vowels.TextGrid
  - ye_formants.Table
  - ye_formants_utf8.csv (UTF-8 format ဖြစ်ရမယ်)
  - ye_vowel_space.png
 
**Note: "ye" ဆိုတဲ့ နေရာမှာ ကိုယ့်နာမည်ကို အစားထိုးပါ။**

## Slide

- Draw Your Own Vowel Chart Slide: [https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/tree/main/slides/6.Draw_Your_Own_Vowel_Space](https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/tree/main/slides/6.Draw_Your_Own_Vowel_Space)  

## 1.Installation

- Praat ကို ကိုယ်စက်မှာ သုံးလို့ရဖို့ installation လုပ်ပါ။  
- Praat Homepage: [https://www.fon.hum.uva.nl/praat/](https://www.fon.hum.uva.nl/praat/)  
- Download Page for Windows OS: [https://www.fon.hum.uva.nl/praat/download_win.html](https://www.fon.hum.uva.nl/praat/download_win.html)

## 2.Learn Shortcuts of Praat

- Praat software က စသုံးစမှာ ခက်နိုင်တာမို့၊ အောက်ပါ shortcut တချို့ကို မှီငြမ်းပါ။

<div align="center">
Table. Essential shortcuts of Praat

| Windows       | Mac            | Function                          |
|--------------|---------------|----------------------------------|
| `Tab`        | `Tab`         | Play from cursor/selected portion |
| `Ctrl+I`     | `Command+I`   | Zoom in                          |
| `Ctrl+O`     | `Command+O`   | Zoom out                         |
| `Ctrl+N`     | `Command+N`   | Zoom to selected portion         |
| `Ctrl+A`     | `Command+A`   | Show all (zoom full range)       |
| `Ctrl+B`     | `Command+B`   | Zoom back (previous zoom level)  |
| `Enter`      | `Return`      | Add boundary at cursor           |
| `Alt+Backspace` | `Option+Delete` | Delete selected boundary        |
| `Ctrl+R`     | `Command+R`   | Run a script                     |
| `Ctrl+T`     | `Command+T`   | Run selected portion of a script |
</div>

## 3.Recording  

- Record Following Myanmar Vowels with Praat (mono, 16 KHz)
   - အ၊ အာ၊ အိ၊ အီ၊ အု၊ အူ၊ အေ၊ အဲ၊ အော
     (သို့မဟုတ်)    
   - အ အာ အိ အု အေ အဲ အော
  

<p align="center">
<img src="https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/code_examples/Draw_Your_Own_Vowel_Space/fig/Recording_with_Praat.png" alt="Recording UI" width="600"/>  
</p>  
<div align="center">
   Fig.1 Recording with Praat (mono, 16KHz)   
</div> 

- Recording လုပ်ထားတဲ့ ဖိုင်ကို list box မှာပြနေပေမဲ့ hard-drive မှာ မသိမ်းရသေးပါဘူး။
- Wave file အဖြစ် save လုပ်ဖို့မမေ့ပါနဲ့။  

## 4. Formant Tracking

- Wave ဖိုင်ကိုဖွင့်ထားပြီး Formants menu အောက်မှာရှိတဲ့ Show formants ကို ရွေးလိုက်ရင် ကွန်ပျူတာက detect လုပ်လို့ရတဲ့ formant line တွေကို အနီရောင်နဲ့ ပြပေးမှာ ဖြစ်ပါတယ်။ အောက်ပါ ဥပမာ ပုံလိုမျိုးပါ။


<p align="center">
<img src="https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/code_examples/Draw_Your_Own_Vowel_Space/fig/5.Check_Formant_Trackings.png" alt="Formant tracking" width="800"/>  
</p>  
<div align="center">
   Fig.1 Formant tracking with Praat
</div> 


## 5.Annotation

- Make Segmentation and Annotation
- Vowel တစ်လုံးချင်းစီအတွက် semgmenation လုပ်ရင်းနဲ့ "အ" ဆိုရင် "a" ၊ "အာ" အသံဆိုရင် "ɑ" ဆိုပြီး text annotation လုပ်သွားပါ။

<p align="center">
<img src="https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/code_examples/Draw_Your_Own_Vowel_Space/fig/14.Annotation.png" alt="Annotation example figure" width="800"/>  
</p>  
<div align="center">
   Fig.1 Annotation Example 
</div> 

- အသံဖမ်းထားတဲ့ vowel အားလုံးကို annotation လုပ်ရမှာဖြစ်ပါတယ်။ ဥပမာ အောက်ပါလိုမျိုး ...

<p align="center">
<img src="https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/code_examples/Draw_Your_Own_Vowel_Space/fig/16.Annotation.png" alt="Fig. Annotation for all vowels" width="800"/>  
</p>  
<div align="center">
   Fig.2 Annotation for all vowels
</div> 

## 5.

## References

1. [Praat Scripting Tutorial](https://www.eleanorchodroff.com/tutorial/PraatScripting.pdf)  
2. [Burmese Phonology](https://en.wikipedia.org/wiki/Burmese_phonology)  
3. [Phonemic Chart](https://englishwithlucy.com/phonemic-chart/)  
4. [https://www.ipachart.com/](https://www.ipachart.com/)  
5. [https://ipa-reader.com/](https://ipa-reader.com/)  
6. [X-ray films of speech](https://www.phonetik.uni-muenchen.de/~hoole/kurse/movies/xray/xray_demo.html)
7. [https://walkergareth.github.io/learnipa/IPAChart/charts/vowels.html](https://walkergareth.github.io/learnipa/IPAChart/charts/vowels.html)  
