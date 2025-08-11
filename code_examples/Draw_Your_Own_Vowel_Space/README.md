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
 
Note: `ye` ဆိုတဲ့ နေရာမှာ `ကိုယ့်နာမည်` ကိုအစားထိုးပါ။

## Slide

- Draw Your Own Vowel Chart Slide: [https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/tree/main/slides/6.Draw_Your_Own_Vowel_Space](https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/tree/main/slides/6.Draw_Your_Own_Vowel_Space)
- ဒီ ဆလိုက်ထဲမှာ လုပ်ရတဲ့ အဆင့်တွေကို screenshot တွေ အများကြီးနဲ့ ရှင်းပြထားပါတယ်။ Seminar ပျက်တဲ့သူတွေက နားမလည်ရင် သူငယ်ချင်းကို မေးတာ၊ တိုင်ပင်တာ လုပ်ပါ။  

## 1. Installation

- Praat ကို ကိုယ့်စက်မှာ သုံးလို့ရဖို့ installation လုပ်ရပါမယ်။  
- Praat Homepage: [https://www.fon.hum.uva.nl/praat/](https://www.fon.hum.uva.nl/praat/)  
- Download Page for Windows OS: [https://www.fon.hum.uva.nl/praat/download_win.html](https://www.fon.hum.uva.nl/praat/download_win.html)

## 2. Learn Shortcuts of Praat

- Praat software က စသုံးစမှာ ခက်နိုင်တာမို့၊ အောက်ပါ shortcut တချို့ကို မှီငြမ်းပါ။

<div align="center">
Table 1. Essential shortcuts of Praat

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

## 3. Recording  

- Record Following Myanmar Vowels with Praat (mono, 16 KHz)
   - အ၊ အာ၊ အိ၊ အီ၊ အု၊ အူ၊ အေ၊ အဲ၊ အော
     (သို့မဟုတ်)    
   - အ အာ အိ အု အေ အဲ အော
  
<br>

<p align="center">
<img src="https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/code_examples/Draw_Your_Own_Vowel_Space/fig/Recording_with_Praat.png" alt="Recording UI" width="500"/>  
</p>  
<div align="center">
   Fig.1 Recording with Praat (mono, 16KHz)   
</div> 

<br>

- Recording လုပ်ထားတဲ့ ဖိုင်ကို list box မှာပြနေပေမဲ့ hard-drive မှာ မသိမ်းရသေးပါဘူး။
- Wave file အဖြစ် save လုပ်ဖို့မမေ့ပါနဲ့။ ဥပမာ။ ။ ye_vowels.wav   

## 4. Formant Tracking

- Wave ဖိုင်ကိုဖွင့်ထားပြီး Formants menu အောက်မှာရှိတဲ့ Show formants ကို ရွေးလိုက်ရင် ကွန်ပျူတာက detect လုပ်လို့ရတဲ့ formant line တွေကို အနီရောင်နဲ့ ပြပေးမှာ ဖြစ်ပါတယ်။ အောက်ပါ ဥပမာ ပုံလိုမျိုးပါ။


<p align="center">
<img src="https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/code_examples/Draw_Your_Own_Vowel_Space/fig/5.Check_Formant_Trackings.png" alt="Formant tracking" width="800"/>  
</p>  
<div align="center">
   Fig.2 Formant tracking with Praat
</div> 

<br>

- Formant (F1, F2, F3) လိုင်းတွေကို Zoom in လုပ်ပြီး vowel တစ်ခုချင်းစီအတွက် စစ်ဆေးပါ။
- ဒီ assignment အလုပ်အတွက်၊ ဆရာတို့က F1, F2, F3 ကိုပဲ အဓိက စိတ်ဝင်စားပါတယ်
- တကယ်လို့ F1, F2, F3 ရဲ့ formant line တွေက တဆက်တည်း မရှိပဲ အရမ်းပြန့်ကျဲနေတာ၊ ပြတ်ထွက်တာတွေ ဖြစ်နေရင်တော့ Formants menu အောက်က Formant settings... ကိုနှိပ်ပြီး ဝင်ညှိရမှာ ဖြစ်ပါတယ်
- နောက်တခု အရေးကြီးတာက ယောက်ျားလေး အသံနဲ့ မိန်းကလေး အသံရဲ့ frequency တွေက မတူပါဘူး။ အခြေခံအားဖြင့် ယောကျ်ားလေးအတွက်က 5000 ပတ်ဝန်းကျင်၊ မိန်းကလေးအတွက်က 5500 ပတ်ဝန်းကျင်ရှိပါတယ်။
  
<p align="center">
<img src="https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/code_examples/Draw_Your_Own_Vowel_Space/fig/8.Formant_Settings.png" alt="Formant setting..." width="800"/>  
</p>  
<div align="center">
   Fig.3 Adjust formant settings based on formant tracking outputs
</div> 

## 5. TextGrid File Preparation

- Segmentation, Annotation မလုပ်ခင်မှာ အရင်ဆုံး TextGrid ဖိုင်ကို ပြင်ဆင်ရပါတယ်
- Praat ရဲ့ list box မှာ ပေါ်နေတဲ့ wavefile ကို selection မှတ်ပြီး၊ ဘေးဘက်က menu တွေထဲက `Annotate` ဆိုတာကို ကလစ်နှိပ်ပြီး sub-menu `To TextGrid...` ဆိုတာကို ရွေးပါ
- Tier ကတော့ တစ်ခုပဲထားပါ။ အော်တိုဖြည့်ပေးထားတာကို ဖျက်လိုက်ပြီး vowels ဆိုပြီး ရိုက်ထည့်ပါ
- `Which of these are point tiers?` ရဲ့ textbox ကိုတော့ blank box အနေနဲ့ပဲ ထားပေးပါ
- အောက်ပါ ပုံတွေကို မှီငြမ်းပါ

<p align="center">
<img src="https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/code_examples/Draw_Your_Own_Vowel_Space/fig/10.Annotate_to_TextGrid.png" alt="Create TextGrid" width="800"/>  
</p>  
<div align="center">
   Fig.4 Create TextGrid file
</div> 

<br>

<p align="center">
<img src="https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/code_examples/Draw_Your_Own_Vowel_Space/fig/11.Tier_Name_No_Point_Tiers.png" alt="Set one tier" width="500"/>  
</p>  
<div align="center">
   Fig.5 Make only one tier
</div> 

## 6. Annotation

- Make Segmentation and Annotation
- Wavefile ရော TextGrid ဖိုင်ရော နှစ်ဖိုင်လုံးကို selection မှတ်ပြီးမှ edit လုပ်ပါ
- Vowel တစ်လုံးချင်းစီအတွက် semgmenation လုပ်ရင်းနဲ့ "အ" ဆိုရင် "a" ၊ "အာ" အသံဆိုရင် "ɑ" ဆိုပြီး text annotation လုပ်သွားပါ။

<br>
<p align="center">
<img src="https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/code_examples/Draw_Your_Own_Vowel_Space/fig/14.Annotation.png" alt="Annotation example figure" width="800"/>  
</p>  
<div align="center">
   Fig.6 Annotation Example 
</div> 

<br>

- အသံဖမ်းထားတဲ့ vowel အားလုံးကို annotation လုပ်ရမှာဖြစ်ပါတယ်။ ဥပမာ အောက်ပါလိုမျိုး ...
<br>

<p align="center">
<img src="https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/code_examples/Draw_Your_Own_Vowel_Space/fig/16.Annotation.png" alt="Fig. Annotation for all vowels" width="800"/>  
</p>  
<div align="center">
   Fig.7 Annotation for all vowels
</div> 

- Vowel အားလုံးကို annotation လုပ်ပြီးတဲ့အခါမှာတော့ TextGrid ဖိုင်ကို နာမည်တစ်ခုပေးပြီး သိမ်းပေးရမှာဖြစ်ပါတယ်။ ဥပမာ ye_vowels.TextGrid

<br>

<p align="center">
<img src="https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/code_examples/Draw_Your_Own_Vowel_Space/fig/18.Save_TextGrid.png" alt="Saving TextGrid file" width="650"/>  
</p>  
<div align="center">
   Fig.8 Saving TextGrid file
</div> 

<br>

- တကယ်လို့ ရှေ့မှာ အသံသွင်းထားတဲ့ wavefile ကို save မလုပ်ရသေးရင်၊ ခု save လုပ်ထားလိုက်ပါဦး။ ဥပမာ။ ။ ye_vowels.wav  

<br>

<p align="center">
<img src="https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/code_examples/Draw_Your_Own_Vowel_Space/fig/17.Save_Wavefile.png" alt="Saving wavefile" width="650"/>  
</p>  
<div align="center">
   Fig.9 Saving wavefile
</div> 

## 7. Estimating Formant Frequencies with Burg Algorithm

- ဒီအဆင့်မှာတော့ wave ဖိုင်ကနေ formant frequency တွေကို Burg algorithm ကို သုံးပြီး ခန့်မှန်းကြည့်မှာဖြစ်ပါတယ်။
- list box ထဲမှာ wave ဖိုင်တစ်ဖိုင်တည်းကိုပဲ ကလစ်လုပ်ပြီး selection မှတ်ပါ
- Selection မှတ်ပြီးသွားရင် `Analyse spectrum` menu ကိုရွေးပြီး `To Formant (burg)...` ကို နှိပ်လိုက်ရင် အောက်ပါအတိုင်း `Sound: To Formant (Burg metho)` ဆိုတဲ့ dialogue box ပေါ်လာပါလိမ့်မယ်
- အဲဒီ dialogue box မှာ `Max. number of formants:` နဲ့ `Formant ceiling (Hz):` တွေကို ရှေ့မှာ Formant tracking လုပ်စဉ်က ပေးထားခဲ့တဲ့ value တွေအတိုင်း ညီအောင်ထားပေးရပါမယ်။ ဆရာအသံအတွက်ကတော့ Formant ceiling ကို `4500.0 Hz` ထားခဲ့ပါတယ်။
- Max. number of formants က ရှေ့မှာလည်း 4.0 ထားခဲ့တာမို့ ခု ဒီ dialogue box မှာလည်း 4.0 ပဲ ထားပါတယ်။

<br>

<p align="center">
<img src="https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/code_examples/Draw_Your_Own_Vowel_Space/fig/20.to_Formant_Same_Settings.png" alt="Setting same values" width="650"/>  
</p>  
<div align="center">
   Fig.10 Setting same values for "Max. number of formants" and "Formant ceiling (Hz)"
</div> 
<br>

- Setting လုပ်ပြီးသွားလို့ OK button ကို နှိပ်လိုက်ရင်တော့ list box မှာ နောက်ထပ် item တစ်ခု တိုးလာတာကို တွေ့ရပါလိမ့်မယ်။
- ပုံမှန်အားဖြင့်ကတော့ wavefile ရဲ့ ဖိုင်နာမည်ကိုပဲ ယူလိုက်ပြီး ရှေ့က Formant တပ်ပေးပါတယ်။ ဥပမာ။ ။ "Formant ye_vowels" လိုနာမည်မျိုးပါ
- အဲလို list item အသစ်တိုးလာပြီးရင်တော့ list box ထဲကနေ နံပါတ် ၁ ဖြစ်တဲ့ wavefile နဲ့ နံပါတ် ၂ ဖြစ်တဲ့ TextGrid ကို ရွေး (i.e. selection လုပ်) ပြီး Praat script ကို run ဖို့ပြင်ကြရအောင်
- `Praat` menu အောက်ကနေ  ` New Praat Script` ဆိုတဲ့ sub-menu ကို နှိပ်လိုက်ပါ
- ပေါ်လာတဲ့ text edit မှာ [extract_formants_from_annotated_textgrids.praat](https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/code_examples/Draw_Your_Own_Vowel_Space/working_with_Praat/extract_formants_from_annotated_textgrids.praat) ဖိုင်ထဲက code တွေကို copy/paste လုပ်ပါ
- Praat Script ထဲက `Name$` variable ကို ကိုယ့်ဖိုင်နာမည်တွေနဲ့ တူတဲ့ နာမည်ကို ပေးရပါတယ်။ ဆရာကတော့ "ye_vowels" လို့ ပေးခဲ့ပါတယ်။\
- ပြီးတော့ `num_timepoints = 10` ဆိုပြီး ထားခဲ့ပါတယ်။ 
- အောက်ပါ ပုံကို မှီငြမ်းပါ

<br>

<p align="center">
<img src="https://github.com/ye-kyaw-thu/LU_Lab_Intern3_2025/blob/main/code_examples/Draw_Your_Own_Vowel_Space/fig/21.Praat_script.png" alt="Praat script for estimating formant frequencies" width="650"/>  
</p>  
<div align="center">
   Fig.11 Praat script for estimating formant frequencies
</div> 
<br>


## References

1. [Praat Scripting Tutorial](https://www.eleanorchodroff.com/tutorial/PraatScripting.pdf)  
2. [Burmese Phonology](https://en.wikipedia.org/wiki/Burmese_phonology)  
3. [Phonemic Chart](https://englishwithlucy.com/phonemic-chart/)  
4. [https://www.ipachart.com/](https://www.ipachart.com/)  
5. [https://ipa-reader.com/](https://ipa-reader.com/)  
6. [X-ray films of speech](https://www.phonetik.uni-muenchen.de/~hoole/kurse/movies/xray/xray_demo.html)
7. [https://walkergareth.github.io/learnipa/IPAChart/charts/vowels.html](https://walkergareth.github.io/learnipa/IPAChart/charts/vowels.html)  
