# Demo of NMT with Marian

ဒီ Practical tutorial က marian NMT framework နဲ့ Transformer model ဆောက်ပြီး၊ မြန်မာ-ရခိုင် neural machine translation လုပ်ပြထားတာပါ။  

y, LU Lab., Myanmar  
29 June 2025  

## Data Information

အရင် ၂၀၁၈တုန်းက YTU မှာ လုပ်ခဲ့တဲ့ Workshop ဒေတာကို ယူသုံးခဲ့တယ်။  
Link: [https://github.com/ye-kyaw-thu/MTRSS](https://github.com/ye-kyaw-thu/MTRSS)    

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data$ wc *.my
   500   3454  57964 dev.my
   100    667  10887 test.my
  5000  33847 561195 train.my
  5600  37968 630046 total
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data$
```

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data$ wc *.rk
   500   3397  57216 dev.rk
   100    661  10633 test.rk
  5000  33302 554426 train.rk
  5600  37360 622275 total
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data$
```

Parallel corpus ကြိုပြင်ထားဖို့ လိုအပ်တယ်။  
စာကြောင်းရေ အရေအတွက်က ညီနေရမယ်။   

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data$ head train.my
မင်း အဲ့ဒါ ကို အခြား တစ်ခုနဲ့ မ ချိတ် ဘူးလား ။
သူမ ဘယ်သူ့ကိုမှ မ မှတ်မိတော့ဘူး ။
အဲ့ဒါ ကျွန်တော်တို့ အတွက် ခက်ခဲတယ် ။
ခင်ဗျား ပြောခဲ့ သလို ကျွန်တော် ရှင်းပြ ခဲ့တယ် ။
သူ့ကို ထိန်းဖို့ မင်း ပဲ တတ်နိုင်တယ် ။
အဲ့ဒါ ကို ကိုယ် တက်နင်း မိသွားလား ။
ငါ စဉ်းစား သလို စဉ်းစားပါ ။
အတင်းပြော ရတာ မုန်းတယ် ။
​နောက်ဆုံး တစ် ကြိမ် သူ့ကို ချစ်ပါတယ် လို့ ပြောခွင့်တောင် မ ရ တော့ဘူး ။
နာဆာ မှ ဒုံးပျံ စတက်တာ နဲ့ သူ မှတ်တမ်း ရေး ခဲ့တယ် ။
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data$
```

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data$ head train.rk
မင်း ယင်းချင့် ကို အခြား တစ်ခုနန့်  မ ချိတ် ပါလား ။
ထိုမချေ   တစ်ယောက်လေ့  မ မှတ်မိပါယာ ။
ယင်းချင့် ကျွန်တော်  ရို့ အတွက် ခက်ခ ရေ ။
မင်း ပြောခ ရေပိုင် ကျွန်တော် ယှင်းပြ ခရေ ။
သူ့ကို ထိန်းဖို့ မင်း ရာ တတ်နိုင်ရေ။
ယင်းချင့် ကို ငါ တက်နင်း မိလား လာ ။
ငါ စဉ်းစား ရေပိုင် စဉ်းစားပါ ။
အတင်းပြော ရစွာ မုန်း ရေ ။
​နောက်ဆုံး တစ် ကြိမ် သူ့ကို ချစ်ပါရေ လို့ ပြောခွင့်တောင် မ ရ ပါ။
နာဆာ မှ ဒုံးပျံ စတက်စွာနန့် သူ မှတ်တမ်း ရွီး ခရေ။
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data$
```
## Marian NMT Framework  

GitHub Link: [https://github.com/marian-nmt/marian](https://github.com/marian-nmt/marian)    
Paper:[https://aclanthology.org/P18-4020/](https://aclanthology.org/P18-4020/)  

## marian Command  

ကိုယ့်စက်ထဲမှာ marian ကို installation လုပ်တာ အဆင်ပြေတယ် ဆိုရင် အောက်ပါလိုမျိုး --help ခေါ်ကြည့်လို့ ရလိမ့်မယ်။  

(base) ye@lst-hpc3090:~/exp/nmt$ marian --help  

```
Marian: Fast Neural Machine Translation in C++
Usage: marian [OPTIONS]

General options:
  -h,--help                             Print this help message and exit
  --version                             Print the version number and exit
  --authors                             Print list of authors and exit
  --cite                                Print citation and exit
  --build-info TEXT                     Print CMake build options and exit. Set to 'all' to print advanced options
  -c,--config VECTOR ...                Configuration file(s). If multiple, later overrides earlier
  -w,--workspace INT=2048               Preallocate arg MB of work space. Negative `--workspace -N` value allocates workspace as total available GPU memory minus N megabytes.
  --log TEXT                            Log training process information to file given by arg
  --log-level TEXT=info                 Set verbosity level of logging: trace, debug, info, warn, err(or), critical, off
  --log-time-zone TEXT                  Set time zone for the date shown on logging
  --quiet                               Suppress all logging to stderr. Logging to files still works
  --quiet-translation                   Suppress logging for translation
  --seed UINT                           Seed for all random number generators. 0 means initialize randomly
  --check-nan                           Check for NaNs or Infs in forward and backward pass. Will abort when found. This is a diagnostic option that will slow down computation significantly
  --interpolate-env-vars                allow the use of environment variables in paths, of the form ${VAR_NAME}
  --relative-paths                      All paths are relative to the config file location
  --dump-config TEXT                    Dump current (modified) configuration to stdout and exit. Possible values: full, minimal, expand
  --sigterm TEXT=save-and-exit          What to do with SIGTERM: save-and-exit or exit-immediately.


Model options:
  -m,--model TEXT=model.npz             Path prefix for model to be saved/resumed. Supported file extensions: .npz, .bin
  --pretrained-model TEXT               Path prefix for pre-trained model to initialize model weights
  --ignore-model-config                 Ignore the model configuration saved in npz file
  --type TEXT=amun                      Model type: amun, nematus, s2s, multi-s2s, transformer
  --dim-vocabs VECTOR=0,0 ...           Maximum items in vocabulary ordered by rank, 0 uses all items in the provided/created vocabulary file
  --dim-emb INT=512                     Size of embedding vector
  --factors-dim-emb INT                 Embedding dimension of the factors. Only used if concat is selected as factors combining form
  --factors-combine TEXT=sum            How to combine the factors and lemma embeddings. Options available: sum, concat
  --lemma-dependency TEXT               Lemma dependency method to use when predicting target factors. Options: soft-transformer-layer, hard-transformer-layer, lemma-dependent-bias, re-embedding
  --lemma-dim-emb INT=0                 Re-embedding dimension of lemma in factors
  --dim-rnn INT=1024                    Size of rnn hidden state
  --enc-type TEXT=bidirectional         Type of encoder RNN : bidirectional, bi-unidirectional, alternating (s2s)
  --enc-cell TEXT=gru                   Type of RNN cell: gru, lstm, tanh (s2s)
  --enc-cell-depth INT=1                Number of transitional cells in encoder layers (s2s)
  --enc-depth INT=1                     Number of encoder layers (s2s)
  --dec-cell TEXT=gru                   Type of RNN cell: gru, lstm, tanh (s2s)
  --dec-cell-base-depth INT=2           Number of transitional cells in first decoder layer (s2s)
  --dec-cell-high-depth INT=1           Number of transitional cells in next decoder layers (s2s)
  --dec-depth INT=1                     Number of decoder layers (s2s)
  --skip                                Use skip connections (s2s)
  --layer-normalization                 Enable layer normalization
  --right-left                          Train right-to-left model
  --input-types VECTOR ...              Provide type of input data if different than 'sequence'. Possible values: sequence, class, alignment, weight. You need to provide one type per input file (if --train-sets) or per TSV field (if --tsv).
  --best-deep                           Use Edinburgh deep RNN configuration (s2s)
  --tied-embeddings                     Tie target embeddings and output embeddings in output layer
  --tied-embeddings-src                 Tie source and target embeddings
  --tied-embeddings-all                 Tie all embedding layers and output layer
  --output-omit-bias                    Do not use a bias vector in decoder output layer
  --transformer-heads INT=8             Number of heads in multi-head attention (transformer)
  --transformer-no-projection           Omit linear projection after multi-head attention (transformer)
  --transformer-rnn-projection          Add linear projection after rnn layer (transformer)
  --transformer-pool                    Pool encoder states instead of using cross attention (selects first encoder state, best used with special token)
  --transformer-dim-ffn INT=2048        Size of position-wise feed-forward network (transformer)
  --transformer-decoder-dim-ffn INT=0   Size of position-wise feed-forward network in decoder (transformer). Uses --transformer-dim-ffn if 0.
  --transformer-ffn-depth INT=2         Depth of filters (transformer)
  --transformer-decoder-ffn-depth INT=0 Depth of filters in decoder (transformer). Uses --transformer-ffn-depth if 0
  --transformer-ffn-activation TEXT=swish
                                        Activation between filters: swish or relu (transformer)
  --transformer-dim-aan INT=2048        Size of position-wise feed-forward network in AAN (transformer)
  --transformer-aan-depth INT=2         Depth of filter for AAN (transformer)
  --transformer-aan-activation TEXT=swish
                                        Activation between filters in AAN: swish or relu (transformer)
  --transformer-aan-nogate              Omit gate in AAN (transformer)
  --transformer-decoder-autoreg TEXT=self-attention
                                        Type of autoregressive layer in transformer decoder: self-attention, average-attention (transformer)
  --transformer-tied-layers VECTOR ...  List of tied decoder layers (transformer)
  --transformer-guided-alignment-layer TEXT=last
                                        Last or number of layer to use for guided alignment training in transformer
  --transformer-preprocess TEXT         Operation before each transformer layer: d = dropout, a = add, n = normalize
  --transformer-postprocess-emb TEXT=d  Operation after transformer embedding layer: d = dropout, a = add, n = normalize
  --transformer-postprocess TEXT=dan    Operation after each transformer layer: d = dropout, a = add, n = normalize
  --transformer-postprocess-top TEXT    Final operation after a full transformer stack: d = dropout, a = add, n = normalize. The optional skip connection with 'a' by-passes the entire stack.
  --transformer-train-position-embeddings
                                        Train positional embeddings instead of using static sinusoidal embeddings
  --transformer-depth-scaling           Scale down weight initialization in transformer layers by 1 / sqrt(depth)
  --bert-mask-symbol TEXT=[MASK]        Masking symbol for BERT masked-LM training
  --bert-sep-symbol TEXT=[SEP]          Sentence separator symbol for BERT next sentence prediction training
  --bert-class-symbol TEXT=[CLS]        Class symbol BERT classifier training
  --bert-masking-fraction FLOAT=0.15    Fraction of masked out tokens during training
  --bert-train-type-embeddings=true     Train bert type embeddings, set to false to use static sinusoidal embeddings
  --bert-type-vocab-size INT=2          Size of BERT type vocab (sentence A and B)
  --dropout-rnn FLOAT                   Scaling dropout along rnn layers and time (0 = no dropout)
  --dropout-src FLOAT                   Dropout source words (0 = no dropout)
  --dropout-trg FLOAT                   Dropout target words (0 = no dropout)
  --transformer-dropout FLOAT           Dropout between transformer layers (0 = no dropout)
  --transformer-dropout-attention FLOAT Dropout for transformer attention (0 = no dropout)
  --transformer-dropout-ffn FLOAT       Dropout for transformer filter (0 = no dropout)


Training options:
  --cost-type TEXT=ce-sum               Optimization criterion: ce-mean, ce-mean-words, ce-sum, perplexity
  --multi-loss-type TEXT=sum            How to accumulate multi-objective losses: sum, scaled, mean
  --unlikelihood-loss                   Use word-level weights as indicators for sequence-level unlikelihood training
  --overwrite                           Do not create model checkpoints, only overwrite main model file with last checkpoint. Reduces disk usage
  --no-reload                           Do not load existing model specified in --model arg
  -t,--train-sets VECTOR ...            Paths to training corpora: source target
  -v,--vocabs VECTOR ...                Paths to vocabulary files have to correspond to --train-sets. If this parameter is not supplied we look for vocabulary files source.{yml,json} and target.{yml,json}. If these files do not exist they are created
  --sentencepiece-alphas VECTOR ...     Sampling factors for SentencePiece vocabulary; i-th factor corresponds to i-th vocabulary
  --sentencepiece-options TEXT          Pass-through command-line options to SentencePiece trainer
  --sentencepiece-max-lines UINT=2000000
                                        Maximum lines to train SentencePiece vocabulary, selected with sampling from all data. When set to 0 all lines are going to be used.
  -e,--after-epochs UINT                Finish after this many epochs, 0 is infinity (deprecated, '--after-epochs N' corresponds to '--after Ne')
  --after-batches UINT                  Finish after this many batch updates, 0 is infinity (deprecated, '--after-batches N' corresponds to '--after Nu')
  -a,--after TEXT=0e                    Finish after this many chosen training units, 0 is infinity (e.g. 100e = 100 epochs, 10Gt = 10 billion target labels, 100Ku = 100,000 updates
  --disp-freq TEXT=1000u                Display information every arg updates (append 't' for every arg target labels)
  --disp-first UINT                     Display information for the first arg updates
  --disp-label-counts=true              Display label counts when logging loss progress
  --save-freq TEXT=10000u               Save model file every arg updates (append 't' for every arg target labels)
  --logical-epoch VECTOR=1e,0 ...       Redefine logical epoch counter as multiple of data epochs (e.g. 1e), updates (e.g. 100Ku) or labels (e.g. 1Gt). Second parameter defines width of fractional display, 0 by default.
  --max-length UINT=50                  Maximum length of a sentence in a training sentence pair
  --max-length-crop                     Crop a sentence to max-length instead of omitting it if longer than max-length
  --tsv                                 Tab-separated input
  --tsv-fields UINT                     Number of fields in the TSV input. By default, it is guessed based on the model type
  --shuffle TEXT=data                   How to shuffle input data (data: shuffles data and sorted batches; batches: data is read in order into batches, but batches are shuffled; none: no shuffling). Use with '--maxi-batch-sort none' in order to achieve exact reading order
  --no-shuffle                          Shortcut for backwards compatiblity, equivalent to --shuffle none (deprecated)
  --no-restore-corpus                   Skip restoring corpus state after training is restarted
  -T,--tempdir TEXT=/tmp                Directory for temporary (shuffled) files and database
  --sqlite TEXT                         Use disk-based sqlite3 database for training corpus storage, default is temporary with path creates persistent storage
  --sqlite-drop                         Drop existing tables in sqlite3 database
  -d,--devices VECTOR=0 ...             Specifies GPU ID(s) to use for training. Defaults to 0..num-devices-1
  --num-devices UINT                    Number of GPUs to use for this process. Defaults to length(devices) or 1
  --no-nccl                             Disable inter-GPU communication via NCCL
  --sharding TEXT=global                When using NCCL and MPI for multi-process training use 'global' (default, less memory usage) or 'local' (more memory usage but faster) sharding
  --sync-freq TEXT=200u                 When sharding is local sync all shards across processes once every n steps (possible units u=updates, t=target labels, e=epochs)
  --cpu-threads UINT=0                  Use CPU-based computation with this many independent threads, 0 means GPU-based computation
  --mini-batch INT=64                   Size of mini-batch used during update
  --mini-batch-words INT                Set mini-batch size based on words instead of sentences
  --mini-batch-fit                      Determine mini-batch size automatically based on sentence-length to fit reserved memory
  --mini-batch-fit-step UINT=10         Step size for mini-batch-fit statistics
  --gradient-checkpointing              Enable gradient-checkpointing to minimize memory usage
  --maxi-batch INT=100                  Number of batches to preload for length-based sorting
  --maxi-batch-sort TEXT=trg            Sorting strategy for maxi-batch: none, src, trg (not available for decoder)
  --shuffle-in-ram                      Keep shuffled corpus in RAM, do not write to temp file
  --data-threads UINT=8                 Number of concurrent threads to use during data reading and processing
  --all-caps-every UINT                 When forming minibatches, preprocess every Nth line on the fly to all-caps. Assumes UTF-8
  --english-title-case-every UINT       When forming minibatches, preprocess every Nth line on the fly to title-case. Assumes English (ASCII only)
  --mini-batch-words-ref UINT           If given, the following hyper parameters are adjusted as-if we had this mini-batch size: --learn-rate, --optimizer-params, --exponential-smoothing, --mini-batch-warmup
  --mini-batch-warmup TEXT=0            Linear ramp-up of MB size, up to this #updates (append 't' for up to this #target labels). Auto-adjusted to --mini-batch-words-ref if given
  --mini-batch-track-lr                 Dynamically track mini-batch size inverse to actual learning rate (not considering lr-warmup)
  --mini-batch-round-up=true            Round up batch size to next power of 2 for more efficient training, but this can make batch size less stable. Disable with --mini-batch-round-up=false
  -o,--optimizer TEXT=adam              Optimization algorithm: sgd, adagrad, adam
  --optimizer-params VECTOR ...         Parameters for optimization algorithm, e.g. betas for Adam. Auto-adjusted to --mini-batch-words-ref if given
  --optimizer-delay FLOAT=1             SGD update delay (#batches between updates). 1 = no delay. Can be fractional, e.g. 0.1 to use only 10% of each batch
  --sync-sgd                            Use synchronous SGD instead of asynchronous for multi-gpu training
  -l,--learn-rate FLOAT=0.0001          Learning rate. Auto-adjusted to --mini-batch-words-ref if given
  --lr-report                           Report learning rate for each update
  --lr-decay FLOAT                      Per-update decay factor for learning rate: lr <- lr * arg (0 to disable)
  --lr-decay-strategy TEXT=epoch+stalled
                                        Strategy for learning rate decaying: epoch, batches, stalled, epoch+batches, epoch+stalled
  --lr-decay-start VECTOR=10,1 ...      The first number of (epoch, batches, stalled) validations to start learning rate decaying (tuple)
  --lr-decay-freq UINT=50000            Learning rate decaying frequency for batches, requires --lr-decay-strategy to be batches
  --lr-decay-reset-optimizer            Reset running statistics of optimizer whenever learning rate decays
  --lr-decay-repeat-warmup              Repeat learning rate warmup when learning rate is decayed
  --lr-decay-inv-sqrt VECTOR=0 ...      Decrease learning rate at arg / sqrt(no. batches) starting at arg (append 't' or 'e' for sqrt(target labels or epochs)). Add second argument to define the starting point (default: same as first value)
  --lr-warmup TEXT=0                    Increase learning rate linearly for arg first batches (append 't' for arg first target labels)
  --lr-warmup-start-rate FLOAT          Start value for learning rate warmup
  --lr-warmup-cycle                     Apply cyclic warmup
  --lr-warmup-at-reload                 Repeat warmup after interrupted training
  --label-smoothing FLOAT               Epsilon for label smoothing (0 to disable)
  --factor-weight FLOAT=1               Weight for loss function for factors (factored vocab only) (1 to disable)
  --clip-norm FLOAT=1                   Clip gradient norm to arg (0 to disable)
  --exponential-smoothing FLOAT=0       Maintain smoothed version of parameters for validation and saving with smoothing factor. 0 to disable. Auto-adjusted to --mini-batch-words-ref if given.
  --guided-alignment TEXT=none          Path to a file with word alignments. Use guided alignment to guide attention or 'none'. If --tsv it specifies the index of a TSV field that contains the alignments (0-based)
  --guided-alignment-cost TEXT=ce       Cost type for guided alignment: ce (cross-entropy), mse (mean square error), mult (multiplication)
  --guided-alignment-weight FLOAT=0.1   Weight for guided alignment cost
  --data-weighting TEXT                 Path to a file with sentence or word weights. If --tsv it specifies the index of a TSV field th
at contains the weights (0-based)
  --data-weighting-type TEXT=sentence   Processing level for data weighting: sentence, word
  --embedding-vectors VECTOR ...        Paths to files with custom source and target embedding vectors
  --embedding-normalization             Normalize values from custom embedding vectors to [-1, 1]
  --embedding-fix-src                   Fix source embeddings. Affects all encoders
  --embedding-fix-trg                   Fix target embeddings. Affects all decoders
  --fp16                                Shortcut for mixed precision training with float16 and cost-scaling, corresponds to: --precision float16 float32 --cost-scaling 8.f 10000 1.f 8.f
  --precision VECTOR=float32,float32 ...
                                        Mixed precision training for forward/backward pass and optimizaton. Defines types for: forward/backward pass, optimization.
  --cost-scaling VECTOR ...             Dynamic cost scaling for mixed precision training: scaling factor, frequency, multiplier, minimum factor
  --gradient-norm-average-window UINT=100
                                        Window size over which the exponential average of the gradient norm is recorded (for logging and scaling). After this many updates about 90% of the mass of the exponential average comes from these updates
  --dynamic-gradient-scaling VECTOR ... Re-scale gradient to have average gradient norm if (log) gradient norm diverges from average by arg1 sigmas. If arg2 = "log" the statistics are recorded for the log of the gradient norm else use plain norm
  --check-gradient-nan                  Skip parameter update in case of NaNs in gradient
  --normalize-gradient                  Normalize gradient by multiplying with no. devices / total labels (not recommended and to be removed in the future)
  --train-embedder-rank VECTOR ...      Override model configuration and train a embedding similarity ranker with the model encoder, parameters encode margin and an optional normalization factor
  --quantize-bits UINT=0                Number of bits to compress model to. Set to 0 to disable
  --quantize-optimization-steps UINT=0  Adjust quantization scaling factor for N steps
  --quantize-log-based                  Uses log-based quantization
  --quantize-biases                     Apply quantization to biases
  --ulr                                 Enable ULR (Universal Language Representation)
  --ulr-query-vectors TEXT              Path to file with universal sources embeddings from projection into universal space
  --ulr-keys-vectors TEXT               Path to file with universal sources embeddings of target keys from projection into universal space
  --ulr-trainable-transformation        Make Query Transformation Matrix A trainable
  --ulr-dim-emb INT                     ULR monolingual embeddings dimension
  --ulr-dropout FLOAT=0                 ULR dropout on embeddings attentions. Default is no dropout
  --ulr-softmax-temperature FLOAT=1     ULR softmax temperature to control randomness of predictions. Deafult is 1.0: no temperature
  --task VECTOR ...                     Use predefined set of options. Possible values: transformer-base, transformer-big, transformer-base-prenorm, transformer-big-prenorm


Validation set options:
  --valid-sets VECTOR ...               Paths to validation corpora: source target
  --valid-freq TEXT=10000u              Validate model every arg updates (append 't' for every arg target labels)
  --valid-metrics VECTOR=cross-entropy ...
                                        Metric to use during validation: cross-entropy, ce-mean-words, perplexity, valid-script, translation, bleu, bleu-detok (deprecated, same as bleu), bleu-segmented, chrf. Multiple metrics can be specified
  --valid-reset-stalled                 Reset stalled validation metrics when the training is restarted
  --valid-reset-all                     Reset all validation metrics when the training is restarted
  --early-stopping UINT=10              Stop if the first validation metric does not improve for arg consecutive validation steps
  --early-stopping-on TEXT=first        Decide if early stopping should take into account first, all, or any validation metricsPossible values: first, all, any
  -b,--beam-size UINT=12                Beam size used during search with validating translator
  -n,--normalize FLOAT=0                Divide translation score by pow(translation length, arg)
  --max-length-factor FLOAT=3           Maximum target length as source length times factor
  --word-penalty FLOAT                  Subtract (arg * translation length) from translation score
  --allow-unk                           Allow unknown words to appear in output
  --n-best                              Generate n-best list
  --word-scores                         Print word-level scores. One score per subword unit, not normalized even if --normalize
  --valid-mini-batch INT=32             Size of mini-batch used during validation
  --valid-max-length UINT=1000          Maximum length of a sentence in a validating sentence pair. Sentences longer than valid-max-length are cropped to valid-max-length
  --valid-script-path TEXT              Path to external validation script. It should print a single score to stdout. If the option is used with validating translation, the output translation file will be passed as a first argument
  --valid-script-args VECTOR ...        Additional args passed to --valid-script-path. These are inserted between the script path and the output translation-file path
  --valid-translation-output TEXT       (Template for) path to store the translation. E.g., validation-output-after-{U}-updates-{T}-tokens.txt. Template parameters: {E} for epoch; {B} for No. of batches within epoch; {U} for total No. of updates; {T} for total No. of tokens seen.
  --keep-best                           Keep best model for each validation metric
  --valid-log TEXT                      Log validation scores to file given by arg

```

## marian-vocab Command

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data$ marian-vocab --help
Create a vocabulary from text corpora given on STDIN
Usage: marian-vocab [OPTIONS]

Allowed options:
  -h,--help                             Print this help message and exit
  --version                             Print the version number and exit
  -m,--max-size UINT=0                  Generate only UINT most common vocabulary items

Examples:
  ./marian-vocab < text.src > vocab.yml
  cat text.src text.trg | ./marian-vocab > vocab.yml
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data$
```

## Concat Train and Dev Files

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data$ mkdir pre
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data$ cat train.my dev.my > ./pre/train_dev.my
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data$ cat train.rk dev.rk > ./pre/train_dev.rk
```

## Build Vocab

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data/pre$ marian-vocab < ./train_dev.my > ../vocab/vocab.my.yml
[2025-06-29 14:59:16] Creating vocabulary...
[2025-06-29 14:59:16] [data] Creating vocabulary stdout from stdin
[2025-06-29 14:59:17] Finished
```

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data/pre$ marian-vocab < ./train_dev.rk > ../vocab/vocab.rk.yml
[2025-06-29 14:59:37] Creating vocabulary...
[2025-06-29 14:59:37] [data] Creating vocabulary stdout from stdin
[2025-06-29 14:59:37] Finished
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data/pre$
```


vocab.my.yml ဖိုင်ထဲမှာက...  

```
</s>: 0
<unk>: 1
။: 2
ကို: 3
မ: 4
ကျွန်တော်: 5
မင်း: 6
သူမ: 7
သူ: 8
ခင်ဗျား: 9
မှာ: 10
တယ်: 11
က: 12
သူတို: 13
ငါ: 14
၊: 15
ဖို: 16
နဲ့: 17
တွေ: 18
ကျွန်တော့်: 19
ရှိ: 20
ဘူး: 21
ဒီ: 22
ဘာ: 23
လို: 24
```

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data/vocab$ head -n 20 ./vocab.rk.yml
</s>: 0
<unk>: 1
။: 2
မင်း: 3
ကို: 4
မ: 5
ကျွန်တော်: 6
သူ: 7
ဖို့: 8
ရေ: 9
ထိုမချေ: 10
ပါ: 11
ငါ: 12
က: 13
သူရို့: 14
လား: 15
တိ: 16
လေး: 17
နန့်: 18
ဇာ: 19
```

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data/vocab$ wc *
  7362  14726 243032 vocab.my.yml
  7829  15660 254540 vocab.rk.yml
 15191  30386 497572 total
```

## Path Information  

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/data/vocab$ pwd
/home/ye/exp/nmt/my-rk/data/vocab
```

## Shell Script Preparation for Training  

(base) ye@lst-hpc3090:~/exp/nmt/my-rk$ cat tf.myrk.sh  

```bash
## Written by Ye Kyaw Thu, LST, NECTEC, Thailand
## Experiments for my-rk demo for Internship3 Students

## Old Notes
#     --mini-batch-fit -w 10000 --maxi-batch 1000 \
#    --mini-batch-fit -w 1000 --maxi-batch 100 \
#     --tied-embeddings-all \
#     --tied-embeddings \
#     --valid-metrics cross-entropy perplexity translation bleu \
#     --transformer-dropout 0.1 --label-smoothing 0.1 \
#     --learn-rate 0.0003 --lr-warmup 16000 --lr-decay-inv-sqrt 16000 --lr-report \
#     --optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \

mkdir -p my-rk/model.tf.myrk;

marian \
    --model my-rk/model.tf.myrk/model.npz --type transformer \
    --train-sets /home/ye/exp/nmt/my-rk/data/train.my \
    /home/ye/exp/nmt/my-rk/data/train.rk \
    --max-length 200 \
    --vocabs /home/ye/exp/nmt/my-rk/data/vocab/vocab.my.yml \
    /home/ye/exp/nmt/my-rk/data/vocab/vocab.rk.yml \
    --mini-batch-fit -w 1000 --maxi-batch 100 \
    --early-stopping 10 \
    --valid-freq 5000 --save-freq 5000 --disp-freq 500 \
    --valid-metrics cross-entropy perplexity bleu \
    --valid-sets /home/ye/exp/nmt/my-rk/data/dev.my \
    /home/ye/exp/nmt/my-rk/data/dev.rk \
    --valid-translation-output my-rk/model.tf.myrk/valid.my-rk.output \
    --quiet-translation \
    --valid-mini-batch 64 \
    --beam-size 6 --normalize 0.6 \
    --log my-rk/model.tf.myrk/train.log \
    --valid-log my-rk/model.tf.myrk/valid.log \
    --enc-depth 2 --dec-depth 2 \
    --transformer-heads 8 \
    --transformer-postprocess-emb d \
    --transformer-postprocess dan \
    --transformer-dropout 0.3 --label-smoothing 0.1 \
    --learn-rate 0.0003 --lr-warmup 0 --lr-decay-inv-sqrt 16000 --lr-report \
    --clip-norm 5 \
    --tied-embeddings \
    --devices 0 --sync-sgd --seed 1111 \
    --exponential-smoothing \
    --dump-config > my-rk/model.tf.myrk/my-rk.config.yml

time marian -c my-rk/model.tf.myrk/my-rk.config.yml  2>&1 | tee my-rk/model.tf.myrk/transformer-myrk.log

```

## Check GPU Status  

GPU အားမအား စစ်ကြည့်ပါ။  

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk$ nvidia-smi
Sun Jun 29 18:15:30 2025
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.230.02             Driver Version: 535.230.02   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090 Ti     Off | 00000000:01:00.0 Off |                  Off |
|  0%   56C    P8              36W / 480W |     59MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A   1718944      G   /usr/lib/xorg/Xorg                           33MiB |
|    0   N/A  N/A   1719135      G   /usr/bin/gnome-shell                         15MiB |
+---------------------------------------------------------------------------------------+
(base) ye@lst-hpc3090:~/exp/nmt/my-rk$
```

ဒီ NMT tutorial အတွက် သုံးခဲ့တဲ့ GPU information ကိုလည်း သိရတာပေါ့။  

## Config File Information

configuration file က tf.myrk.sh ကို run လိုက်ရင် ဆောက်ပေးပါလိမ့်မယ်။  
အဲဒီ config ဖိုင်ထဲမှာ ပေးထားတဲ့ parameter, path စတဲ့ အချက်အလက်တွေနဲ့ marian က NMT model ဆောက်ပေးမှာပါ။  

(base) ye@lst-hpc3090:~/exp/nmt/my-rk/my-rk/model.tf.myrk$ cat my-rk.config.yml  

```
# Marian configuration file generated at 2025-06-29 16:02:10 +0700 with version v1.12.0 65bf82ff 2023-02-21 09:56:29 -0800
# General options
authors: false
cite: false
build-info: ""
workspace: 1000
log: my-rk/model.tf.myrk/train.log
log-level: info
log-time-zone: ""
quiet: false
quiet-translation: true
seed: 1111
check-nan: false
interpolate-env-vars: false
relative-paths: false
sigterm: save-and-exit
# Model options
model: my-rk/model.tf.myrk/model.npz
pretrained-model: ""
ignore-model-config: false
type: transformer
dim-vocabs:
  - 0
  - 0
dim-emb: 512
factors-dim-emb: 0
factors-combine: sum
lemma-dependency: ""
lemma-dim-emb: 0
dim-rnn: 1024
enc-type: bidirectional
enc-cell: gru
enc-cell-depth: 1
enc-depth: 2
dec-cell: gru
dec-cell-base-depth: 2
dec-cell-high-depth: 1
dec-depth: 2
skip: false
layer-normalization: false
right-left: false
input-types:
  []
best-deep: false
tied-embeddings: true
tied-embeddings-src: false
tied-embeddings-all: false
output-omit-bias: false
transformer-heads: 8
transformer-no-projection: false
transformer-rnn-projection: false
transformer-pool: false
transformer-dim-ffn: 2048
transformer-decoder-dim-ffn: 0
transformer-ffn-depth: 2
transformer-decoder-ffn-depth: 0
transformer-ffn-activation: swish
transformer-dim-aan: 2048
transformer-aan-depth: 2
transformer-aan-activation: swish
transformer-aan-nogate: false
transformer-decoder-autoreg: self-attention
transformer-tied-layers:
  []
transformer-guided-alignment-layer: last
transformer-preprocess: ""
transformer-postprocess-emb: d
transformer-postprocess: dan
transformer-postprocess-top: ""
transformer-train-position-embeddings: false
transformer-depth-scaling: false
bert-mask-symbol: "[MASK]"
bert-sep-symbol: "[SEP]"
bert-class-symbol: "[CLS]"
bert-masking-fraction: 0.15
bert-train-type-embeddings: true
bert-type-vocab-size: 2
dropout-rnn: 0
dropout-src: 0
dropout-trg: 0
transformer-dropout: 0.3
transformer-dropout-attention: 0
transformer-dropout-ffn: 0
# Training options
cost-type: ce-sum
multi-loss-type: sum
unlikelihood-loss: false
overwrite: false
no-reload: false
train-sets:
  - /home/ye/exp/nmt/my-rk/data/train.my
  - /home/ye/exp/nmt/my-rk/data/train.rk
vocabs:
  - /home/ye/exp/nmt/my-rk/data/vocab/vocab.my.yml
  - /home/ye/exp/nmt/my-rk/data/vocab/vocab.rk.yml
sentencepiece-alphas:
  []
sentencepiece-options: ""
sentencepiece-max-lines: 2000000
after-epochs: 0
after-batches: 0
after: 0e
disp-freq: 500
disp-first: 0
disp-label-counts: true
save-freq: 5000
logical-epoch:
  - 1e
  - 0
max-length: 200
max-length-crop: false
tsv: false
tsv-fields: 0
shuffle: data
no-shuffle: false
no-restore-corpus: false
tempdir: /tmp
sqlite: ""
sqlite-drop: false
devices:
  - 0
num-devices: 0
no-nccl: false
sharding: global
sync-freq: 200u
cpu-threads: 0
mini-batch: 64
mini-batch-words: 0
mini-batch-fit: true
mini-batch-fit-step: 10
gradient-checkpointing: false
maxi-batch: 100
maxi-batch-sort: trg
shuffle-in-ram: false
data-threads: 8
all-caps-every: 0
english-title-case-every: 0
mini-batch-words-ref: 0
mini-batch-warmup: 0
mini-batch-track-lr: false
mini-batch-round-up: true
optimizer: adam
optimizer-params:
  []
optimizer-delay: 1
sync-sgd: true
learn-rate: 0.0003
lr-report: true
lr-decay: 0
lr-decay-strategy: epoch+stalled
lr-decay-start:
  - 10
  - 1
lr-decay-freq: 50000
lr-decay-reset-optimizer: false
lr-decay-repeat-warmup: false
lr-decay-inv-sqrt:
  - 16000
lr-warmup: 0
lr-warmup-start-rate: 0
lr-warmup-cycle: false
lr-warmup-at-reload: false
label-smoothing: 0.1
factor-weight: 1
clip-norm: 5
exponential-smoothing: 0.0001
guided-alignment: none
guided-alignment-cost: ce
guided-alignment-weight: 0.1
data-weighting: ""
data-weighting-type: sentence
embedding-vectors:
  []
embedding-normalization: false
embedding-fix-src: false
embedding-fix-trg: false
fp16: false
precision:
  - float32
  - float32
cost-scaling:
  []
gradient-norm-average-window: 100
dynamic-gradient-scaling:
  []
check-gradient-nan: false
normalize-gradient: false
train-embedder-rank:
  []
quantize-bits: 0
quantize-optimization-steps: 0
quantize-log-based: false
quantize-biases: false
ulr: false
ulr-query-vectors: ""
ulr-keys-vectors: ""
ulr-trainable-transformation: false
ulr-dim-emb: 0
ulr-dropout: 0
ulr-softmax-temperature: 1
task:
  []
# Validation set options
valid-sets:
  - /home/ye/exp/nmt/my-rk/data/dev.my
  - /home/ye/exp/nmt/my-rk/data/dev.rk
ulr-keys-vectors: ""
ulr-trainable-transformation: false
ulr-dim-emb: 0
ulr-dropout: 0
ulr-softmax-temperature: 1
task:
  []
# Validation set options
valid-sets:
  - /home/ye/exp/nmt/my-rk/data/dev.my
  - /home/ye/exp/nmt/my-rk/data/dev.rk
valid-freq: 5000
valid-metrics:
  - cross-entropy
  - perplexity
  - bleu
valid-reset-stalled: false
valid-reset-all: false
early-stopping: 10
early-stopping-on: first
beam-size: 6
normalize: 0.6
max-length-factor: 3
word-penalty: 0
allow-unk: false
n-best: false
word-scores: false
valid-mini-batch: 64
valid-max-length: 1000
valid-script-path: ""
valid-script-args:
  []
valid-translation-output: my-rk/model.tf.myrk/valid.my-rk.output
keep-best: false
valid-log: my-rk/model.tf.myrk/valid.log

```

## Training  

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk$ ./tf.myth.zen.chk1.sh   
...
...
...
[2025-06-29 16:17:37] Ep. 2392 : Up. 55000 : Sen. 1,488 : Cost 1.23307192 * 833,705 @ 1,714 after 91,591,963 : Time 8.26s : 100921.98 words/s : gNorm 0.0925 : L.r. 1.6181e-04
[2025-06-29 16:17:37] Saving model weights and runtime parameters to my-rk/model.tf.myrk/model.iter55000.npz
[2025-06-29 16:17:37] Saving model weights and runtime parameters to my-rk/model.tf.myrk/model.npz
[2025-06-29 16:17:38] Saving Adam parameters
[2025-06-29 16:17:38] [training] Saving training checkpoint to my-rk/model.tf.myrk/model.npz and my-rk/model.tf.myrk/model.npz.optimizer.npz
[2025-06-29 16:17:39] [valid] Ep. 2392 : Up. 55000 : cross-entropy : 27.0907 : stalled 10 times (last best: 23.8792)
[2025-06-29 16:17:39] [valid] Ep. 2392 : Up. 55000 : perplexity : 32.3252 : stalled 10 times (last best: 21.4085)
[2025-06-29 16:17:39] [valid] Ep. 2392 : Up. 55000 : bleu : 35.1594 : stalled 10 times (last best: 36.7707)
[2025-06-29 16:17:39] Training finished
[2025-06-29 16:17:39] Saving model weights and runtime parameters to my-rk/model.tf.myrk/model.npz
[2025-06-29 16:17:39] Saving Adam parameters
[2025-06-29 16:17:39] [training] Saving training checkpoint to my-rk/model.tf.myrk/model.npz and my-rk/model.tf.myrk/model.npz.optimizer.npz

real    15m30.410s
user    16m1.097s
sys     0m36.928s
(base) ye@lst-hpc3090:~/exp/nmt/my-rk$
```

training data က စာကြောင်းရေ ၅၀၀၀ နဲ့ပဲမို့ လက်ရှိ စက်မှာ မြန်ပါတယ်။  

## Check validation file During Training  

Current Hypothesis File:  

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/my-rk/model.tf.myrk$ head ./valid.my-rk.output
ကျွန်တော် နက်ဖန် ကား အသစ် တိ ကြည့် ဖို့လို့ ။
မင်း ဇာ တိ သတင်းပီး ဖို့လေး။
အကြံဉာဏ် လိုချင် လား ။
မင်း ဇာ တိ ထမ်း နီစွာလေး ။
မင်း ငါ့ကို မ မုန်း ခပါ နောက် ၊ မုန်း ခပါလား ။
အပြင်မှာ ရေပိုင်ယာ ။
ကျွန်တော် မင်းကို အိမ် ဖြစ်ပါရေ။
မင်း မီးဖွင့် ပါ ။
ဇာသူ အလုပ် တော့ခါ ဖို့လေး။
ကျွန်တော်လည်း ဒေအတိုင်း လုပ်မိဖို့ ။
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/my-rk/model.tf.myrk$
```

Reference File:  

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/my-rk/model.tf.myrk$ head ../../data/dev.rk
ကျွန်တော် နက်ဖန် ကား အသစ် တိ လား ကြည့် ဖို့လို့ ။
မင်း ဇာ တိ သတင်းပီး ဖို့လေး။
အကြံဉာဏ် ကောင်းတိ လိုချင် လား ။
မင်း ဇာ တိ သယ် နီစွာလေး ။
မင်း ငါ့ကို မ မုန်း ခပါ  နောက် ၊ မုန်း ခပါလား ။
အပြင်မှာ မှောင်နီ သီးရေ ။
ကျွန်တော် ခင်ဗျားကို အိမ် လိုက်ပို့ပီးမေ ။
မင်း ကတိမပျက် ပါ ။
ဇာသူ အလုပ် လာလုပ် ဖို့လေး။
ငါလည်း ယင်းပိုင် ထင်ရေ ။
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/my-rk/model.tf.myrk$
```

## Check Training Output Folder

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/my-rk$ tree ./model.tf.myrk/
./model.tf.myrk/
├── model.iter10000.npz
├── model.iter15000.npz
├── model.iter20000.npz
├── model.iter25000.npz
├── model.iter30000.npz
├── model.iter35000.npz
├── model.iter40000.npz
├── model.iter45000.npz
├── model.iter50000.npz
├── model.iter5000.npz
├── model.iter55000.npz
├── model.npz
├── model.npz.decoder.yml
├── model.npz.optimizer.npz
├── model.npz.progress.yml
├── model.npz.yml
├── my-rk.config.yml
├── train.log
├── transformer-myrk.log
├── valid.log
└── valid.my-rk.output

1 directory, 21 files
```

## Preparing Shell Script for Testing

(base) ye@lst-hpc3090:~/exp/nmt/my-rk/my-rk/model.tf.myrk$ cat test-eval.sh  

```bash
#!/bin/bash

for i in {5000..55000..5000}
do
    marian-decoder -m ./model.iter$i.npz -v /home/ye/exp/nmt/my-rk/data/vocab/vocab.my.yml /home/ye/exp/nmt/my-rk/data/vocab/vocab.rk.yml --devices 0 --output hyp.iter$i.rk < /home/ye/exp/nmt/my-rk/data/test.my;
    echo "Evaluation with hyp.iter$i.th, Transformer model:" >> eval-result.txt;
    perl /home/ye/tool/mosesbin/ubuntu-17.04/moses/scripts/generic/multi-bleu.perl /home/ye/exp/nmt/my-rk/data/test.rk < ./hyp.iter$i.rk >> eval-result.txt;

done
```

## Testing  

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/my-rk/model.tf.myrk$ ./test-eval.sh
...
...
...
[2025-06-29 16:42:05] Best translation 82 : ကျေးဇူးပြုပြီး ကျွန်တော် သသုံး ဖောင်သာ ဟိလေး ။
[2025-06-29 16:42:05] Best translation 83 : သူ မလာ ကျလေး မ ကျ ပါ ။
[2025-06-29 16:42:05] Best translation 84 : ဒေ အမှတ်တံဆိပ် ထထိုင်ခခုံ မင်း ဝီစု ကူညီ နနိုင်ဖဖိဖို့လား ။
[2025-06-29 16:42:05] Best translation 85 : မင်းရရဲ့ လည်ချောင်းကကို ကြည့်ပါရစီ။
[2025-06-29 16:42:05] Best translation 86 : ကျွန်တော် ယင်းချင့်ကကို မ မြင် လလိုက်ပါ ။
[2025-06-29 16:42:05] Best translation 87 : မင်း သူရရိရို့ ကကို သနား ဖဖိဖို့လား ။
[2025-06-29 16:42:05] Best translation 88 : မင်း ညွှန်ကြား ရေ အဖြေ ။
[2025-06-29 16:42:05] Best translation 89 : ယင်းချင့်ကကို ကျေးဇူးပြုပြီး လိကျင့် ပီးပါ ။
[2025-06-29 16:42:05] Best translation 90 : အချေတိ သကြားလလုံး ကြြိုက် နီရေ ။
[2025-06-29 16:42:05] Best translation 91 : မိန့်ပစ် ဖဖိဖို့ ခက်ပါ ရေ ။
[2025-06-29 16:42:05] Best translation 92 : သသုံးလေးပတ် ကြာပြီးခါ ကကိုယ်စားလှယ် ကကို မာန်ပါ ဖဖိဖို့ဗျာလ်ထော ။
[2025-06-29 16:42:05] Best translation 93 : ငါရရိရို့ စောစော ပြန် ကတ်မယ် ။
[2025-06-29 16:42:05] Best translation 94 : ကားခ ဇာလောက် ကျ ဖဖိဖို့ ခန့်မှန်း လေး ။
[2025-06-29 16:42:05] Best translation 95 : အဂု မင်း ဇာ လား ဖဖိဖို့လေး။
[2025-06-29 16:42:05] Best translation 96 : ငါရရိရို့ မင်း ကကို ပြော လလိလို့လား ။
[2025-06-29 16:42:05] Best translation 97 : ​ကောင်းစွာ က ကကိုယ် ကကောင်း အလုပ်များ လလိလို့ပါ ။
[2025-06-29 16:42:05] Best translation 98 : ချစ်မြတ်နနိုးသူ ကောင်းကောင်း ကြာ ဗျာယ် ။
[2025-06-29 16:42:05] Best translation 99 : ချစ်မြတ်နနိုးသူ မင်း လာ နနိုင်ရေ ။
[2025-06-29 16:42:05] Total time: 0.73303s wall
ERROR: could not find reference file /home/ye/exp/nmt/my-th/data/test.rk at /home/ye/tool/mosesbin/ubuntu-17.04/moses/scripts/generic/multi-bleu.perl line 32.
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/my-rk/model.tf.myrk$
```

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/my-rk/model.tf.myrk$ ./test-eval.sh
...
...
...
[2025-06-29 16:49:52] Best translation 67 : ငါ တစ်ည လုံး စာကျက် နီရေ ။
[2025-06-29 16:49:52] Best translation 68 : သုံးလေးပတ် ကြာပြီးခါ ကိုယ်စားလှယ် ဖြစ် ရဖို့ ။
[2025-06-29 16:49:52] Best translation 69 : သူရို့ကို ဇာသူရို့ ချစ်ကတ်လေး ။
[2025-06-29 16:49:52] Best translation 70 : ဒေပိုင် စိုင်းစားကေ့ တော့ မကြားရစွာ နှစ်ပတ်လောက်ဟိဗျာယ် ။
[2025-06-29 16:49:52] Best translation 71 : ယင်း အိမ် ဖြစ်ပါရေ။
[2025-06-29 16:49:52] Best translation 72 : သူ့ကို မိန်း သင့် သလောက် မိန်းပါ ။
[2025-06-29 16:49:52] Best translation 73 : ငါ မင်းကို မ ကူညီ နိုင်ပါ ။
[2025-06-29 16:49:52] Best translation 74 : ထိုမချေ ငါ့ ကို မ မှတ်မိ ပါ ။
[2025-06-29 16:49:52] Best translation 75 : အဂု အလုပ် မ လုပ် ကတ်လား ။
[2025-06-29 16:49:52] Best translation 76 : ထိုမချေ ငါ့ကို ဇာပိုင်နည်း နန့် အပြစ်တင်လေး ။
[2025-06-29 16:49:52] Best translation 77 : မင်း ဇာကြောင့် ရန်ဖြစ် နီ လေး ဆိုစွာ ကျွန်တော် မ မိန်း ချင် ပါ ။
[2025-06-29 16:49:52] Best translation 78 : ကျွန်တော်ရို့ စာမိန်းပွဲ အောင် ရေ ။
[2025-06-29 16:49:52] Best translation 79 : ကောင်မချေ က ကောင်းကောင်း လုပ်ရေ ။
[2025-06-29 16:49:52] Best translation 80 : ကိုယ် စနေနိမာ ဖုန်းဆက်မယ် ။
[2025-06-29 16:49:52] Best translation 81 : ကိုယ်က လားချိုးပါ ။
[2025-06-29 16:49:52] Best translation 82 : ကျေးဇူးပြုပြီး ကျွန်တော် သုံး ဖောင်သာ ဟိလေး ။
[2025-06-29 16:49:52] Best translation 83 : သူ မလာ ကျလေး မ ကျ ပါ ။
[2025-06-29 16:49:52] Best translation 84 : ဒေ အမှတ်တံဆိပ် ထိုင်ခုံ မင်း ဝီစု ကူညီ နိုင်ဖို့လား ။
[2025-06-29 16:49:52] Best translation 85 : မင်းရဲ့ လည်ချောင်းကို ကြည့်ပါရစီ။
[2025-06-29 16:49:52] Best translation 86 : ကျွန်တော် ယင်းချင့်ကို မ မြင် လိုက်ပါ ။
[2025-06-29 16:49:52] Best translation 87 : မင်း သူရို့ ကို သနား ဖို့လား ။
[2025-06-29 16:49:52] Best translation 88 : မင်း ညွှန်ကြား ရေ အဖြေ ။
[2025-06-29 16:49:52] Best translation 89 : ယင်းချင့်ကို ကျေးဇူးပြုပြီး လိကျင့် ပီးပါ ။
[2025-06-29 16:49:52] Best translation 90 : အချေတိ သကြားလုံး ကြိုက် နီရေ ။
[2025-06-29 16:49:52] Best translation 91 : မိန့်ပစ် ဖို့ ခက်ပါ ရေ ။
[2025-06-29 16:49:52] Best translation 92 : သုံးလေးပတ် ကြာပြီးခါ ကိုယ်စားလှယ် ကို မာန်ပါ ဖို့ဗျာလ်ထော ။
[2025-06-29 16:49:52] Best translation 93 : ငါရို့ စောစော ပြန် ကတ်မယ် ။
[2025-06-29 16:49:52] Best translation 94 : ကားခ ဇာလောက် ကျ ဖို့ ခန့်မှန်း လေး ။
[2025-06-29 16:49:52] Best translation 95 : အဂု မင်း ဇာ လား ဖို့လေး။
[2025-06-29 16:49:52] Best translation 96 : ငါရို့ မင်း ကို ပြော လို့လား ။
[2025-06-29 16:49:52] Best translation 97 : ​ကောင်းစွာ က ကိုယ် ကကောင်း အလုပ်များ လို့ပါ ။
[2025-06-29 16:49:52] Best translation 98 : ချစ်မြတ်နိုးသူ ကောင်းကောင်း ကြာ ဗျာယ် ။
[2025-06-29 16:49:52] Best translation 99 : ချစ်မြတ်နိုးသူ မင်း လာ နိုင်ရေ ။
[2025-06-29 16:49:52] Total time: 0.72603s wall
```

## Check Model Folder After Testing  

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/my-rk/model.tf.myrk$ ls
eval-result.txt   hyp.iter40000.rk     model.iter20000.npz  model.iter5000.npz       my-rk.config.yml
hyp.iter10000.rk  hyp.iter45000.rk     model.iter25000.npz  model.iter55000.npz      test-eval.sh
hyp.iter15000.rk  hyp.iter50000.rk     model.iter30000.npz  model.npz                train.log
hyp.iter20000.rk  hyp.iter5000.rk      model.iter35000.npz  model.npz.decoder.yml    transformer-myrk.log
hyp.iter25000.rk  hyp.iter55000.rk     model.iter40000.npz  model.npz.optimizer.npz  valid.log
hyp.iter30000.rk  model.iter10000.npz  model.iter45000.npz  model.npz.progress.yml   valid.my-rk.output
hyp.iter35000.rk  model.iter15000.npz  model.iter50000.npz  model.npz.yml
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/my-rk/model.tf.myrk$
```

Check BLEU Scores ...  

```
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/my-rk/model.tf.myrk$ cat ./eval-result.txt
Evaluation with hyp.iter5000.th, Transformer model:
BLEU = 44.08, 72.0/52.6/44.2/38.0 (BP=0.878, ratio=0.885, hyp_len=585, ref_len=661)
Evaluation with hyp.iter10000.th, Transformer model:
BLEU = 44.51, 72.2/53.3/44.0/37.5 (BP=0.887, ratio=0.893, hyp_len=590, ref_len=661)
Evaluation with hyp.iter15000.th, Transformer model:
BLEU = 43.69, 71.6/52.2/43.6/37.1 (BP=0.882, ratio=0.888, hyp_len=587, ref_len=661)
Evaluation with hyp.iter20000.th, Transformer model:
BLEU = 43.52, 71.0/51.5/41.6/35.1 (BP=0.905, ratio=0.909, hyp_len=601, ref_len=661)
Evaluation with hyp.iter25000.th, Transformer model:
BLEU = 42.26, 70.6/50.3/41.0/34.1 (BP=0.895, ratio=0.900, hyp_len=595, ref_len=661)
Evaluation with hyp.iter30000.th, Transformer model:
BLEU = 43.72, 70.9/51.9/42.5/36.4 (BP=0.895, ratio=0.900, hyp_len=595, ref_len=661)
Evaluation with hyp.iter35000.th, Transformer model:
BLEU = 43.24, 70.8/51.2/41.5/34.9 (BP=0.903, ratio=0.908, hyp_len=600, ref_len=661)
Evaluation with hyp.iter40000.th, Transformer model:
BLEU = 43.79, 71.4/52.6/43.4/37.4 (BP=0.882, ratio=0.888, hyp_len=587, ref_len=661)
Evaluation with hyp.iter45000.th, Transformer model:
BLEU = 43.04, 70.8/51.5/41.6/34.2 (BP=0.902, ratio=0.906, hyp_len=599, ref_len=661)
Evaluation with hyp.iter50000.th, Transformer model:
BLEU = 44.21, 71.5/51.8/41.8/35.9 (BP=0.910, ratio=0.914, hyp_len=604, ref_len=661)
Evaluation with hyp.iter55000.th, Transformer model:
BLEU = 44.88, 72.8/53.5/44.3/38.6 (BP=0.883, ratio=0.890, hyp_len=588, ref_len=661)
(base) ye@lst-hpc3090:~/exp/nmt/my-rk/my-rk/model.tf.myrk$
```

တခြား NMT Framework တွေလည်း ရှိပါသေးတယ်။ ဥပမာ [OpenNMT](https://github.com/OpenNMT/OpenNMT-py), [FairSeq](https://github.com/facebookresearch/fairseq) တို့နဲ့ပါ မော်ဒယ်တွေ ဆောက်ကြည့်ပြီး NMT ကို လေ့လာသွားကြပါလို့။  
