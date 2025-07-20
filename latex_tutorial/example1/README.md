# Notes


ဒီ example ကို run ဖို့အတွက် လိုအပ်တာက intro_latex.tex, references.bib နှစ်ဖိုင်ပါပဲ။  
သို့သော်လည်း run ရတာ အဆင်ပြေဖို့အတွက် shell script (compile.sh) ကို ဥပမာအနေနဲ့ ရေးပြထားပါတယ်။  
လက်တွေ့ ကိုယ့် terminal မှာ စာတမ်းဘာညာကို ရေးတဲ့အခါမှာလည်း စာတမ်းကို ပြင်လိုက် PDF ဖိုင် ထုတ်ကြည့်လိုက်၊ ပြန်ပြင်လိုက် စသည်ဖြင့် အကြိမ်ကြိမ်အခါခါ run ကြရတာမို့ အခုလိုမျိုး shell script ရေးထားတာက ပိုပြီး လုပ်ရကိုင်ရ အဆင်ပြေလို့ပါ။  

## Install Xelatex and Fonts

```bash
sudo apt install texlive-xetex
```

OR  

including Fonts:  

```bash
sudo apt install texlive-xetex texlive-fonts-recommended texlive-fonts-extra fonts-noto
```

## Install biblatex

If you're using a TeX Live system (most likely on Linux), you can install biblatex via the terminal:  

```bash
sudo tlmgr install biblatex
```

## Check Myanmar Fonts

```bash
fc-list | grep -i "myanmar\|padauk"
```

## How to Compile

```bash
pdflatex intro_latex_example.tex
biber intro_latex_example
pdflatex intro_latex_example.tex
pdflatex intro_latex_example.tex
```

OR  

For your convenience, I wrote a shell script and you can run as follows:  

```bash
./compile.sh intro_latex
```
