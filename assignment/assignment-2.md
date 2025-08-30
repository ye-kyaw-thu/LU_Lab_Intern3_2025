# Assignment-2

LU Lab. Internship ၃ လအတွင်းမှာ ဒုတိယမြောက် assignment ပါ။

## Measuring Cartoon Understanding with LLM

(1). ဒီတခါ assignment က "စာမပါတဲ့ ကာတွန်း ၁၂ ပုံ" ကိုစိတ်ကြိုက် ရှာပြီးတော့ LLM model သုံးခုကို (ဥပမာ chatGPT, Gemini, Claude) အောက်ပါ prompt နဲ့ ကာတွန်းပုံကို ဘယ်လောက် နားလည်သလဲ ဆိုတာကို မေးကြည့်ရအောင်။  

Role: You are a professional cartoon caption writer and multilingual visual content analyst.
Task: You are given a cartoon image. Generate two outputs in four languages (English, Japanese, Chinese, Myanmar):

1. Caption: A short, witty, and creative caption (≤12 words).  
2. Description: A concise, neutral summary (2 to 3 sentences) explaining what is happening in the cartoon.  

Important Rules:

- Provide the same meaning for captions and descriptions across all four languages.
- Avoid cultural references that may not translate well.
- Keep JSON output strictly valid and consistent.

JSON Output Format: 

{
  "caption": {
    "en": "English caption",
    "ja": "Japanese caption",
    "zh": "Chinese caption",
    "my": "Myanmar caption"
  },
  "description": {
    "en": "English description",
    "ja": "Japanese description",
    "zh": "Chinese description",
    "my": "Myanmar description"
  }
}

ကျောင်းသား အားလုံးက ဒီ prompt ကိုပဲ သုံးပါ။ ဘာမှ အပြောင်းအလဲ မလုပ်ပါနဲ့။  

(2). LLM model က ထွက်လာတဲ့ အဖြေတွေကို မော်ဒယ် တစ်ခုစီအတွက် ဖိုင်တစ်ခုစီသိမ်းပါ။

(3). Caption နဲ့ description တွေကို ဘယ်လို evaluation ကို ဘယ်လိုလုပ်မလဲဆိုတာကို လေ့လာပြီး evaluation လုပ်ပါ။ Evaluation method က တစ်မျိုးထက်မက သုံးနိုင်တယ်။ ပြီးရင်တော့ ရလာတဲ့ score တွေကို LLM မော်ဒယ် သုံးခုအကြား နှိုင်းယှဉ်ပြီး report တင်ပါ။

(4). ကိုယ်သုံးခဲ့တဲ့ ပုံတွေကော၊ LLM or Visual-LM တွေက ရလာတဲ့ ရလဒ်တွေကော၊ ပြီးတော့ evaluation result report ဖိုင်တွေကောကို folder တစ်ခုအောက်ထဲမှာ သိမ်းဆည်းပါ။ ဥပမာ အောက်ပါလိုမျိုး

cartoon_understanding/
├── cartoons/
│   ├── C001.jpg
│   ├── C002.jpg
│   ├── C003.jpg
│   ├── C004.jpg
│   ├── C005.jpg
│   ├── C006.jpg
│   ├── C007.jpg
│   ├── C008.jpg
│   ├── C009.jpg
│   ├── C0010.jpg
├── evaluation.xlsx  
├── prompt.txt
├── chatGPT.txt
├── Gemini.txt
└── Claude.txt

ကာတွန်းပုံတွေကို ရွေးတဲ့အခါမှာ မြန်မာကာတွန်းဆရာတွေ ဆွဲထားတဲ့ပုံတွေကိုလည်း သုံးလို့ ရပါတယ်။ စာတော့ မပါပါစေနဲ့။ 
အထက်ပါ ဖိုင်တွေအပြင် Evaluation လုပ်တဲ့အခါမှာ သုံးခဲ့တဲ့နည်းလမ်းက ကိုယ့်ဖာသာကိုယ် စဉ်းစားထားတဲ့ နည်းလမ်းအသစ်တခုဆိုရင် အဲဒါနဲ့ ပတ်သက်ပြီး ဖြည့်ရှင်းပြထားတဲ့ ဖိုင်ကိုလည်း ဖြည့်စွက်ပါ။  

(5). ဒီ assignment မှာ အဓိက အလေ့အကျင့်ရစေချင်တာက၊ စဉ်းစားစေချင်တာက ပုံတွေကို ဘယ်လို scenario နဲ့ ရွေးချယ်မှာလဲ၊ သုံးမယ့် LLM or VLM တွေကိုကော ဘယ်အချက်တွေပေါ်မူတည်ပြီး ဆုံးဖြတ်မှာလဲ၊ evaluation method ကကော ဘယ်လို လုပ်မှာလဲ ဆိုတဲ့အချက်တွေပါ။ 

(6). Evaluation အပိုင်းက ဆရာ လုပ်ပြမထားပေမဲ့ နားလည်ရလွယ်အောင် ကာတွန်းပုံ သုံးပုံကို သတ်မှတ်ထားတဲ့ prompt ကို သုံးပြီး chatGPT, Gemini နဲ့ Claude ကနေ reply ပြန်လာတဲ့ output တွေကို ဥပမာအနေနဲ့ ပို့ပေးထားပါမယ်။  

ကြိုးစားပြီး လုပ်ကြပါ။  


y
26 Aug 2025
