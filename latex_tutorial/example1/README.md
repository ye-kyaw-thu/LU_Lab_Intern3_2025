# Notes

## Install Xelatex and Fonts

sudo apt install texlive-xetex

OR

including Fonts:  
sudo apt install texlive-xetex texlive-fonts-recommended texlive-fonts-extra fonts-noto
 
## Install biblatex

If you're using a TeX Live system (most likely on Linux), you can install biblatex via the terminal:
sudo tlmgr install biblatex

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
