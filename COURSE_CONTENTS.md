# PyTorch Course Contents and Practical Reference

This index is organized for two kinds of lookup:

1. **I need to implement something** - start with [Find by task](#find-by-task).
2. **I want to revisit a lesson** - use the [course map](#course-map).

Links point to the repository's Python copies rather than notebooks. Links to named functions and classes include line anchors, so they open close to the implementation in VS Code and repository viewers.

## Find by task

| Goal | Best starting points |
|---|---|
| Build a custom `Dataset` | [PlantsDataset](Course1/Module3/Ass1/ass1.py#L30), [FlowerDataset](Course1/Module3/Lab1/lab1.py#L103), [TextDataset](Course2/Module3/Lab3/lab3.py#L207), [TranslationDataset](Course3/Module3/Assignment1/assignment1.py#L239) |
| Split data and use different train/validation transforms | [SubsetWithTransform](Course1/Module3/Ass1/ass1.py#L275), [get_dataloaders](Course1/Module3/Ass1/ass1.py#L293), [create_dataset_splits](Course2/Module2/Assignment1/assignment1.py#L37) |
| Tune `DataLoader` performance | [worker experiments](Course2/Module4/Lab1/lab1.py#L34), [batch-size experiments](Course2/Module4/Lab1/lab1.py#L129), [pinned-memory experiments](Course2/Module4/Lab1/lab1.py#L202) |
| Compute normalization statistics | [image dataset mean/std](Course1/Module3/Ass1/ass1.py#L181), [torchvision mean/std](Course2/Module2/Lab1/lab1.py#L246) |
| Add image augmentation | [flower augmentation](Course1/Module3/Lab1/lab1.py#L455), [CIFAR transformations](Course1/Module4/Ass1/c1_m4_ass1.py#L31), [transfer-learning transformations](Course2/Module2/Assignment1/assignment1.py#L91) |
| Write a reusable training/evaluation loop | [dense classifier](Course1/Module2/Lab1/lab1.py#L153), [CNN loop](Course1/Module4/Ass1/c1_m4_ass1.py#L495), [shared Siamese loops](Course3/Module1/Lab1/training_functions.py#L1) |
| Handle class imbalance and choose metrics | [classification metrics](Course2/Module1/Lab1/lab1.py#L251), [class weights](Course2/Module3/Assignment1/assignment1.py#L323), [weighted contrastive loss](Course3/Module1/Lab1/lab1b.py#L289) |
| Tune hyperparameters | [Optuna objective](Course2/Module1/Lab3/lab3.py#L137), [configurable search space](Course2/Module1/Assignment1/assignment1.py#L204) |
| Schedule the learning rate | [scheduler comparison](Course2/Module1/Lab2/lab2.py#L244), [Lightning plateau scheduler](Course2/Module4/Assignment1/assignment1.py#L317) |
| Select a model using accuracy, size, and speed | [efficiency measurement](Course2/Module1/Lab4/lab4.py#L221), [weighted selection](Course2/Module1/Lab4/lab4.py#L276) |
| Use transfer learning or fine-tuning | [MobileNetV3 setup](Course2/Module2/Assignment1/assignment1.py#L214), [feature extraction vs fine-tuning](Course2/Module2/Lab4/lab4.py#L1), [partial BERT freezing](Course2/Module3/Assignment1/assignment1.py#L380) |
| Profile and accelerate training | [Lightning profiler](Course2/Module4/Lab2/lab2.py#L1), [mixed precision and accumulation](Course2/Module4/Lab3/lab3.py#L298) |
| Track experiments | [MLflow callback](Course3/Module4/Lab1/lab1.py#L361) |
| Inspect a CNN or explain predictions | [activation inspection](Course1/Module4/Lab2/lab2.py#L1), [saliency](Course3/Module2/Lab2/lab2.py#L103), [Grad-CAM](Course3/Module2/Lab2/lab2.py#L265) |
| Build text classifiers | [EmbeddingBag classifier](Course2/Module3/Lab3/lab3.py#L368), [DistilBERT dataset](Course2/Module3/Assignment1/assignment1.py#L116), [Transformer sentiment model](Course3/Module3/Lab2/lab2.py#L548) |
| Build a Transformer | [manual self-attention](Course3/Module3/Lab1/lab1.py#L357), [encoder-decoder translation](Course3/Module3/Assignment1/assignment1.py#L719), [decoder-only generation](Course3/Module3/Lab3/lab3.py#L376) |
| Use embeddings or metric learning | [GloVe and `nn.Embedding`](Course2/Module3/Lab2/lab2.py#L1), [Siamese network](Course3/Module1/Assignment1/assignment1.py#L766) |
| Deploy or reduce a model | [ONNX conversion](Course3/Module4/Lab2/lab2.py#L1), [pruning](Course3/Module4/Assignment1/assignment1.py#L262), [quantization-aware training](Course3/Module4/Assignment1/assignment1.py#L691) |

## Course map

### Course 1: PyTorch Fundamentals

#### Module 1: Getting Started with PyTorch

| Material | Dataset and model | Techniques and key APIs | Source landmarks |
|---|---|---|---|
| Lab 1: Building a Simple Neural Network | Inline delivery data; single `nn.Linear` neuron | Tensors, linear regression, MSE, SGD, `backward`, `step`, `zero_grad`, `torch.no_grad` | [lab1.py](Course1/Module1/Lab1/lab1.py) |
| Lab 2: Modeling Non-Linear Patterns | Inline delivery data; 1-3-1 ReLU MLP | Standardization, de-standardization, `nn.Sequential`, `nn.ReLU`, loss plotting, inference | [lab2.py](Course1/Module1/Lab2/lab2.py) |
| Lab 3: Tensors: The Core of PyTorch | `data.csv`, pandas/NumPy/synthetic data; no model | Tensor creation, shape, dtype, indexing, masks, reshape, transpose, concatenate, broadcasting, reductions, casting | [lab3.py](Course1/Module1/Lab3/lab3.py) |
| Assignment: Deeper Regression, Smarter Features | Delivery-time CSV; 4-64-32-1 MLP | Feature engineering, pandas-to-tensor pipeline, normalization, train/test masks, regression training and prediction | [rush_hour_feature](Course1/Module1/Assignment/c1_m1_ass1.py#L54), [prepare_data](Course1/Module1/Assignment/c1_m1_ass1.py#L89), [init_model](Course1/Module1/Assignment/c1_m1_ass1.py#L193), [train_model](Course1/Module1/Assignment/c1_m1_ass1.py#L241) |

#### Module 2: The PyTorch Workflow

| Material | Dataset and model | Techniques and key APIs | Source landmarks |
|---|---|---|---|
| Lab 1: Building Your First Image Classifier | MNIST; dense `SimpleMNISTDNN` | `Compose`, `ToTensor`, `Normalize`, `DataLoader`, Adam, cross-entropy, device selection, training/evaluation, prediction visualization | [SimpleMNISTDNN](Course1/Module2/Lab1/lab1.py#L101), [train_epoch](Course1/Module2/Lab1/lab1.py#L153), [evaluate](Course1/Module2/Lab1/lab1.py#L239) |
| Assignment: EMNIST Letter Detective | EMNIST Letters; dense classifier | Image orientation correction, loaders, training, total and per-class accuracy, hidden-message decoding | [create_emnist_dataloaders](Course1/Module2/Ass1/c1_m2_ass1.py#L100), [initialize_emnist_model](Course1/Module2/Ass1/c1_m2_ass1.py#L150), [train_epoch](Course1/Module2/Ass1/c1_m2_ass1.py#L200), [evaluate](Course1/Module2/Ass1/c1_m2_ass1.py#L312), [decode_word_imgs](Course1/Module2/Ass1/c1_m2_ass1.py#L460) |

#### Module 3: Data Management in PyTorch

| Material | Dataset and model | Techniques and key APIs | Source landmarks |
|---|---|---|---|
| Lab 1: Data Management | Oxford 102 Flowers and corrupted copy; no model | Custom `Dataset`, PIL loading, train/validation/test splitting, resize/crop/normalize, augmentation, robust loading, error logging and access monitoring | [FlowerDataset](Course1/Module3/Lab1/lab1.py#L103), [split_dataset](Course1/Module3/Lab1/lab1.py#L371), [get_augmentation_transform](Course1/Module3/Lab1/lab1.py#L455), [RobustFlowerDataset](Course1/Module3/Lab1/lab1.py#L608), [MonitoredDataset](Course1/Module3/Lab1/lab1.py#L846) |
| Assignment: Building a Robust Data Pipeline | Plants classification data; no model | CSV labels, custom `Dataset`, mean/std calculation, augmentation, `Subset`, split-specific transforms and loaders | [PlantsDataset](Course1/Module3/Ass1/ass1.py#L30), [get_mean_std](Course1/Module3/Ass1/ass1.py#L181), [get_transformations](Course1/Module3/Ass1/ass1.py#L217), [SubsetWithTransform](Course1/Module3/Ass1/ass1.py#L275), [get_dataloaders](Course1/Module3/Ass1/ass1.py#L293) |

#### Module 4: Core Neural Network Components

| Material | Dataset and model | Techniques and key APIs | Source landmarks |
|---|---|---|---|
| Lab 1: Building a CNN for Nature Classification | 9- and 15-class CIFAR-100 subsets; custom CNN | Convolution, ReLU, max pooling, dropout, augmentation, architecture/data-flow inspection, scaling class count | [SimpleCNN](Course1/Module4/Lab1/lab1.py#L68), [print_data_flow](Course1/Module4/Lab1/lab1.py#L156), [training_loop](Course1/Module4/Lab1/lab1.py#L211) |
| Lab 2: Debugging Neural Networks | Local image data; broken/fixed CNNs and SqueezeNet | Shape tracing, diagnosing missing flattening, sequential refactoring, activations, module and parameter inspection | [SimpleCNNDebug](Course1/Module4/Lab2/lab2.py#L58), [SimpleCNNFixed](Course1/Module4/Lab2/lab2.py#L112), [SimpleCNN2SeqDebug](Course1/Module4/Lab2/lab2.py#L208) |
| Assignment: Modular Nature Classifier | Selected CIFAR-100 classes; modular CNN | Conv-BatchNorm-ReLU blocks, pooling, dropout, augmentation, train/validation loops, best-state retention | [define_transformations](Course1/Module4/Ass1/c1_m4_ass1.py#L31), [CNNBlock](Course1/Module4/Ass1/c1_m4_ass1.py#L148), [SimpleCNN](Course1/Module4/Ass1/c1_m4_ass1.py#L227), [training_loop](Course1/Module4/Ass1/c1_m4_ass1.py#L495) |

### Course 2: PyTorch Techniques and Ecosystem Tools

#### Module 1: Hyperparameter Optimization

| Material | Dataset and model | Techniques and key APIs | Source landmarks |
|---|---|---|---|
| Lab 1: Tuning and Metrics | Balanced/imbalanced CIFAR-10; `SimpleCNN` | Learning-rate comparison, Accuracy, F1, Precision, Recall, macro averaging | [SimpleCNN](Course2/Module1/Lab1/lab1.py#L23), [evaluate_accuracy](Course2/Module1/Lab1/lab1.py#L70), [train_and_evaluate_metrics](Course2/Module1/Lab1/lab1.py#L251) |
| Lab 2: Learning-Rate Schedulers | CIFAR-10; `SimpleCNN` | `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, scheduler placement and LR histories | [SimpleCNN](Course2/Module1/Lab2/lab2.py#L47), [scheduler comparison](Course2/Module1/Lab2/lab2.py#L244) |
| Lab 3: Optuna | CIFAR-10 and apple anomaly data; flexible CNNs | TPE and grid samplers, search spaces, studies, trials, histories, parallel-coordinate and importance plots | [FlexibleCNN](Course2/Module1/Lab3/lab3.py#L26), [objective](Course2/Module1/Lab3/lab3.py#L137), [FlexibleSimpleCNN](Course2/Module1/Lab3/lab3.py#L237), [objective_apples](Course2/Module1/Lab3/lab3.py#L296) |
| Lab 4: Model Efficiency and Selection | CIFAR-10; OptimizedCNN and ResNet-34 | Parameter/model-size measurement, timed inference, validation split, weighted and constraint-based selection | [architectures](Course2/Module1/Lab4/model_architectures.py), [BasicBlock](Course2/Module1/Lab4/model_architectures.py#L66), [ResNet34](Course2/Module1/Lab4/model_architectures.py#L120), [evaluate_efficiency](Course2/Module1/Lab4/lab4.py#L221), [select_best_model_weighted](Course2/Module1/Lab4/lab4.py#L276) |
| Assignment: Architecture and Hyperparameter Search | AI-vs-real images; configurable `FlexibleCNN` | Optuna search/fixed trials, accuracy/precision/recall, parameter count, inference efficiency, weighted and alternative objectives | [FlexibleCNN](Course2/Module1/Assignment1/assignment1.py#L40), [design_search_space](Course2/Module1/Assignment1/assignment1.py#L204), [objective_function](Course2/Module1/Assignment1/assignment1.py#L286), [add_efficiency_metrics](Course2/Module1/Assignment1/assignment1.py#L469) |

#### Module 2: Working with Images Using TorchVision

| Material | Dataset and model | Techniques and key APIs | Source landmarks |
|---|---|---|---|
| Lab 1: TorchVision Preprocessing | Local images and Oxford-IIIT Pet; no model | PIL-to-tensor and back, grids, save image, resize/crop/flip/jitter, custom salt-and-pepper noise, normalization | [SaltAndPepperNoise](Course2/Module2/Lab1/lab1.py#L179), [calculate_mean_std](Course2/Module2/Lab1/lab1.py#L246) |
| Lab 2: TorchVision Datasets | CIFAR-10, EMNIST Digits, ImageFolder fruit/vegetables, FakeData, FashionMNIST, SVHN | Built-in/custom dataset loading, transforms, `DataLoader`, batch visualization | [lab2.py](Course2/Module2/Lab2/lab2.py) |
| Lab 3: Pretrained Vision Tasks | ImageNet-style samples; ResNet-50, DeepLabV3-ResNet50, Faster R-CNN ResNet50-FPN | Weights metadata/preprocessing, softmax/top-k classification, semantic masks, object boxes | [class metadata](Course2/Module2/Lab3/lab3.py#L64), [classification/segmentation/detection workflow](Course2/Module2/Lab3/lab3.py), [detect_and_draw_bboxes](Course2/Module2/Lab3/lab3.py#L292) |
| Lab 4: Transfer Learning | Image folders and EMNIST Digits; MobileNetV3-Small and ResNet-18 | Feature extraction vs partial fine-tuning, freezing/unfreezing, replacing classifier heads | [lab4.py](Course2/Module2/Lab4/lab4.py) |
| Assignment: AI-vs-Real Transfer Learning | AI-vs-real images; pretrained MobileNetV3-Large | `ImageFolder`, random split, augmentation, local weights, classifier-head adaptation | [create_dataset_splits](Course2/Module2/Assignment1/assignment1.py#L37), [define_transformations](Course2/Module2/Assignment1/assignment1.py#L91), [load_mobilenetv3_model](Course2/Module2/Assignment1/assignment1.py#L214) |

#### Module 3: Working with Text Using Hugging Face

| Material | Dataset and model | Techniques and key APIs | Source landmarks |
|---|---|---|---|
| Lab 1: Tokenization | Inline sentences; BERT tokenizer | Manual vocabulary/token IDs, OOV handling, BERT subwords, `BertTokenizerFast`, `AutoTokenizer` | [tokenize](Course2/Module3/Lab1/lab1.py#L21), [build_vocab](Course2/Module3/Lab1/lab1.py#L40) |
| Lab 2: Word Embeddings | GloVe 6B 100d and synthetic pairs; `SimpleEmbeddingModel` | Cosine similarity, nearest neighbors, analogies, PCA, `nn.Embedding`, training embeddings from scratch | [find_closest_words](Course2/Module3/Lab2/lab2.py#L31), [SimpleEmbeddingModel](Course2/Module3/Lab2/lab2.py#L257), [training_loop](Course2/Module3/Lab2/lab2.py#L311) |
| Lab 3: Text Classification from Scratch | Fruit/vegetable recipe data; EmbeddingBag and manual-pooling classifiers | Cleaning, vocabulary, custom dataset/collators, offsets, class weights, pooled embeddings | [Vocabulary](Course2/Module3/Lab3/lab3.py#L135), [TextDataset](Course2/Module3/Lab3/lab3.py#L207), [EmbeddingBagClassifier](Course2/Module3/Lab3/lab3.py#L368), [ManualPoolingClassifier](Course2/Module3/Lab3/lab3.py#L419) |
| Lab 4: Fine-Tuned Transformer Classification | Recipe data; DistilBERT | Hugging Face tokenization, custom dataset/loaders, full vs partial fine-tuning, class-weighted loss | [RecipeDataset](Course2/Module3/Lab4/lab4.py#L99), [lab4.py](Course2/Module3/Lab4/lab4.py) |
| Assignment: Instruction Classification | Augmented Databricks Dolly 15K; DistilBERT sequence classifier | Dynamic padding, data splits, class weighting, partial layer freezing | [InstructionDataset](Course2/Module3/Assignment1/assignment1.py#L116), [create_data_collator](Course2/Module3/Assignment1/assignment1.py#L211), [calculate_class_weights](Course2/Module3/Assignment1/assignment1.py#L323), [partially_freeze_bert_layers](Course2/Module3/Assignment1/assignment1.py#L380) |

#### Module 4: Efficient Training Pipelines

| Material | Dataset and model | Techniques and key APIs | Source landmarks |
|---|---|---|---|
| Lab 1: DataLoader Optimization | CIFAR-10; no focus model | `num_workers`, batch size, `pin_memory`, `prefetch_factor`, timing, cached experiment results | [workers](Course2/Module4/Lab1/lab1.py#L34), [batch sizes](Course2/Module4/Lab1/lab1.py#L129), [pinned memory](Course2/Module4/Lab1/lab1.py#L202), [custom_experiment](Course2/Module4/Lab1/lab1.py#L348) |
| Lab 2: Profiling | CIFAR-10; configurable Lightning CNN | `PyTorchProfiler`, profiler schedules, memory profiling, baseline/efficient model comparison | [CIFAR10DataModule](Course2/Module4/Lab2/lab2.py#L38), [CIFAR10LightningModule](Course2/Module4/Lab2/lab2.py#L99) |
| Lab 3: Training Optimization | CIFAR-10; Lightning CNN | Mixed precision, gradient accumulation, batch-size trade-offs, CUDA peak memory and timing | [PerformanceCallback](Course2/Module4/Lab3/lab3.py#L227), [run_training](Course2/Module4/Lab3/lab3.py#L298), [run_optimization](Course2/Module4/Lab3/lab3.py#L334) |
| Assignment: Lightning Chest X-Ray Classifier | Chest X-rays; pretrained ResNet-18 | `LightningDataModule`, `LightningModule`, augmentation, AdamW, plateau scheduler, metrics, early stopping | [ChestXRayDataModule](Course2/Module4/Assignment1/assignment1.py#L131), [ChestXRayClassifier](Course2/Module4/Assignment1/assignment1.py#L317), [early_stopping](Course2/Module4/Assignment1/assignment1.py#L452), [run_training](Course2/Module4/Assignment1/assignment1.py#L505) |

### Course 3: Advanced PyTorch

#### Module 1: Advanced CNN Architectures and Metric Learning

| Material | Dataset and model | Techniques and key APIs | Source landmarks |
|---|---|---|---|
| Lab 1: Siamese Networks | Signatures and LEVIR-CD-derived change images; Siamese networks | Pair/triplet sampling, contrastive loss, imbalance weighting, schedulers, one-shot verification and change detection | [SignatureTripletDataset](Course3/Module1/Lab1/lab1a.py#L40), [ChangeDetectionDataset](Course3/Module1/Lab1/lab1b.py#L121), [WeightedContrastiveLoss](Course3/Module1/Lab1/lab1b.py#L289), [training functions](Course3/Module1/Lab1/training_functions.py) |
| Lab 2: Residual Learning | Aerial Landscapes; plain CNN vs custom ResNet | Residual shortcuts, projection/downsampling, augmentation, `torchinfo`, comparative training | [PlainBlock](Course3/Module1/Lab2/lab2.py#L85), [PlainCNN](Course3/Module1/Lab2/lab2.py#L128), [ResidualBlock](Course3/Module1/Lab2/lab2.py#L224), [SimpleResNet](Course3/Module1/Lab2/lab2.py#L315) |
| Lab 3: DenseNet | UCMerced Land Use; custom DenseNet and DenseNet-121 | Dense feature concatenation/reuse, growth rate, transition layers, augmentation, pretrained comparison | [DenseLayer](Course3/Module1/Lab3/lab3.py#L99), [DenseBlock](Course3/Module1/Lab3/lab3.py#L199), [TransitionLayer](Course3/Module1/Lab3/lab3.py#L298), [DenseNet](Course3/Module1/Lab3/lab3.py#L365) |
| Assignment: Efficient CNNs and Metric Learning | Small clothing dataset; MobileNet-like and Siamese models | Inverted residuals, depthwise convolution, triplet sampling, embedding retrieval, Adam/AdamW, StepLR | [InvertedResidualBlock](Course3/Module1/Assignment1/assignment1.py#L93), [MobileNetBackbone](Course3/Module1/Assignment1/assignment1.py#L275), [TripleDataset](Course3/Module1/Assignment1/assignment1.py#L551), [SiameseNetwork](Course3/Module1/Assignment1/assignment1.py#L766) |

#### Module 2: CNN Interpretability and Generative Models

| Material | Dataset and model | Techniques and key APIs | Source landmarks |
|---|---|---|---|
| Lab 1: Features and Receptive Fields | Sample/ImageNet images; three-layer CNN and ResNet-50 | Convolution/pooling visualization, forward hooks, feature maps, effective receptive fields | [ThreeLayerCNN](Course3/Module2/Lab1/lab1.py#L120), [grab hook](Course3/Module2/Lab1/lab1.py#L187) |
| Lab 2: Saliency and Grad-CAM | ImageNet samples; ResNet-50 | Input gradients, forward/backward hooks, activation-gradient weighting, heatmap overlays | [compute_saliency_map](Course3/Module2/Lab2/lab2.py#L103), [GradCAM](Course3/Module2/Lab2/lab2.py#L265), [compute_gradcam](Course3/Module2/Lab2/lab2.py#L355) |
| Lab 3: Diffusion Pipelines | Generated images; Stable Diffusion 2 Base and Google DDPM Bedroom | Seeds/generators, prompts, negative prompts, guidance scale, denoising callbacks, image similarity | [load_model_pipeline](Course3/Module2/Lab3/lab3.py#L53), [save_intermediate_steps](Course3/Module2/Lab3/lab3.py#L293), [get_closest_img](Course3/Module2/Lab3/lab3.py#L451) |
| Assignment: Explainability and Synthetic Data | Healthy/rotten fruit; ResNet-50 and Stable Diffusion 2 Base | Feature hierarchy, saliency, simplified CAM, deterministic generation, denoising visualization | [cnn_feature_hierarchy](Course3/Module2/Assignment1/assignment1.py#L151), [saliency_map](Course3/Module2/Assignment1/assignment1.py#L409), [simplified_cam](Course3/Module2/Assignment1/assignment1.py#L560), [load_sd_pipeline](Course3/Module2/Assignment1/assignment1.py#L723), [denoising_movie](Course3/Module2/Assignment1/assignment1.py#L1000) |

#### Module 3: Transformers

| Material | Dataset and model | Techniques and key APIs | Source landmarks |
|---|---|---|---|
| Lab 1: Self-Attention | Tiny examples and Tiny Shakespeare; manual attention, MHA and next-word model | Vocabulary/tokenization, scaled dot-product attention, positional embeddings, `nn.MultiheadAttention`, autoregressive sampling | [ManualSelfAttention](Course3/Module3/Lab1/lab1.py#L357), [SelfAttnWithMHA](Course3/Module3/Lab1/lab1.py#L995), [NextWordGenerator](Course3/Module3/Lab1/lab1.py#L1172) |
| Lab 2: Transformer Encoders | IMDB reviews; custom and PyTorch encoder classifiers | Padding masks, positional encoding, attention blocks, `TransformerEncoder`, sentiment classification | [EncoderBlock](Course3/Module3/Lab2/lab2.py#L50), [IMDBDataset](Course3/Module3/Lab2/lab2.py#L445), [custom model](Course3/Module3/Lab2/lab2.py#L548), [PyTorch encoder model](Course3/Module3/Lab2/lab2.py#L996) |
| Lab 3: Decoder-Only Generation | Tiny Shakespeare; custom decoder/generator | Causal and padding masks, positional encoding, decoder blocks, autoregressive generation | [make_causal_mask](Course3/Module3/Lab3/lab3.py#L112), [DecoderBlock](Course3/Module3/Lab3/lab3.py#L250), [Decoder](Course3/Module3/Lab3/lab3.py#L376), [ShakespeareGenerator](Course3/Module3/Lab3/lab3.py#L573) |
| Assignment: Machine Translation | English paired with six languages; custom encoder-decoder Transformer | Vocabulary/special tokens, padding and causal masks, positional encoding, encoder/decoder layers, decoding and sampling | [TranslationDataset](Course3/Module3/Assignment1/assignment1.py#L239), [Encoder](Course3/Module3/Assignment1/assignment1.py#L474), [Decoder](Course3/Module3/Assignment1/assignment1.py#L609), [EncoderDecoder](Course3/Module3/Assignment1/assignment1.py#L719), [translate_sentence](Course3/Module3/Assignment1/assignment1.py#L874) |

#### Module 4: Production, Tracking, and Model Optimization

| Material | Dataset and model | Techniques and key APIs | Source landmarks |
|---|---|---|---|
| Lab 1: Experiment Tracking | CIFAR-10; Lightning `SimpleCNN` | MLflow runs, parameters, metrics, artifacts, model logging, callbacks, confusion matrices | [CIFAR10DataModule](Course3/Module4/Lab1/lab1.py#L85), [SimpleCNN](Course3/Module4/Lab1/lab1.py#L177), [MLflowLoggingCallback](Course3/Module4/Lab1/lab1.py#L361) |
| Lab 2: ONNX and TensorFlow Conversion | 28-class fruit/vegetable disease subset; ResNet-18 | `torch.onnx.export`, ONNX validation, ONNX Runtime, `onnx_tf.prepare`, TensorFlow SavedModel inference | [lab2.py](Course3/Module4/Lab2/lab2.py) |
| Lab 3: Pruning | Fruit/vegetable disease subset; demo CNN and ResNet-18 | Unstructured, structured and global pruning, masks, `prune.remove`, sparsity/performance analysis | [SimpleModel](Course3/Module4/Lab3/lab3.py#L43), [analyze_model_sparsity](Course3/Module4/Lab3/lab3.py#L495) |
| Lab 4: Quantization | CIFAR-10; baseline, static-quantized and QAT CNNs | Dynamic/static quantization, observers, calibration, module fusion, `prepare`, `convert`, QAT | [CNN](Course3/Module4/Lab4/lab4.py#L48), [QuantizedCNN](Course3/Module4/Lab4/lab4.py#L252), [calibrate](Course3/Module4/Lab4/lab4.py#L386), [QATCNN](Course3/Module4/Lab4/lab4.py#L495) |
| Assignment: Edge Deployment | Three-class CleanStreet images; ResNet-18 variants | L1/structured pruning, dynamic INT8 quantization, fusion, quantization-aware training, latency/checkpoint comparison | [prune_model](Course3/Module4/Assignment1/assignment1.py#L262), [quantize_dynamic_linear](Course3/Module4/Assignment1/assignment1.py#L417), [fuse_model_inplace](Course3/Module4/Assignment1/assignment1.py#L520), [QATWrapper](Course3/Module4/Assignment1/assignment1.py#L635), [prepare_qat](Course3/Module4/Assignment1/assignment1.py#L691) |

## Dataset index

| Domain | Datasets | Where used |
|---|---|---|
| Tabular/regression | Delivery-time CSV, inline delivery data | Course 1 Module 1 |
| Handwritten characters | MNIST, EMNIST Letters/Digits, FashionMNIST | Course 1 Module 2; Course 2 Module 2 |
| General image classification | CIFAR-10, CIFAR-100, FakeData, SVHN | Course 1 Module 4; Course 2 Modules 1, 2 and 4; Course 3 Module 4 |
| Plants, flowers, food and disease | Oxford 102 Flowers, Oxford-IIIT Pet, plants, fruit/vegetables, apple anomaly, healthy/rotten fruit, plant disease | Course 1 Module 3; Course 2 Modules 1-3; Course 3 Modules 2 and 4 |
| People and generated images | CelebA/COCO/VOC/ImageNet catalog examples, AI-vs-real images | Course 2 Modules 1 and 2 |
| Remote sensing/change detection | LEVIR-CD-derived changes, Aerial Landscapes, UCMerced Land Use | Course 3 Module 1 |
| Medical and environmental | Chest X-rays, CleanStreet | Course 2 Module 4; Course 3 Module 4 |
| Text | Databricks Dolly 15K, recipe text, GloVe 6B, IMDB, Tiny Shakespeare, multilingual translation pairs | Course 2 Module 3; Course 3 Module 3 |
| Generative | Stable Diffusion 2 Base, Google DDPM Bedroom | Course 3 Module 2 |

## Model and architecture index

| Family | Models and concepts | Most useful examples |
|---|---|---|
| Dense networks | Linear regression, MLPs, MNIST/EMNIST dense classifiers | [first neuron](Course1/Module1/Lab1/lab1.py), [deeper regression](Course1/Module1/Assignment/c1_m1_ass1.py#L193), [MNIST DNN](Course1/Module2/Lab1/lab1.py#L101) |
| CNN fundamentals | Conv2d, pooling, batch normalization, dropout, modular blocks | [CNNBlock](Course1/Module4/Ass1/c1_m4_ass1.py#L148), [CNN debugging](Course1/Module4/Lab2/lab2.py#L58) |
| Residual/dense/efficient CNNs | ResNet-18/34/50, DenseNet-121, MobileNetV3 Small/Large, SqueezeNet | [custom ResNet](Course3/Module1/Lab2/lab2.py#L315), [custom DenseNet](Course3/Module1/Lab3/lab3.py#L365), [MobileNet backbone](Course3/Module1/Assignment1/assignment1.py#L275) |
| Pretrained vision tasks | ResNet-50 classification, DeepLabV3 segmentation, Faster R-CNN detection | [pretrained task workflow](Course2/Module2/Lab3/lab3.py) |
| Metric learning | Siamese networks, embeddings, contrastive loss, triplet sampling | [Siamese assignment](Course3/Module1/Assignment1/assignment1.py#L766), [weighted contrastive loss](Course3/Module1/Lab1/lab1b.py#L289) |
| Text models | `nn.Embedding`, `EmbeddingBag`, DistilBERT | [embedding model](Course2/Module3/Lab2/lab2.py#L257), [EmbeddingBag classifier](Course2/Module3/Lab3/lab3.py#L368), [DistilBERT assignment](Course2/Module3/Assignment1/assignment1.py) |
| Transformers | Manual attention, multi-head attention, encoder, decoder, encoder-decoder | [attention](Course3/Module3/Lab1/lab1.py#L357), [sentiment encoder](Course3/Module3/Lab2/lab2.py#L548), [translation model](Course3/Module3/Assignment1/assignment1.py#L719) |
| Generative models | Stable Diffusion 2, DDPM | [diffusion lab](Course3/Module2/Lab3/lab3.py), [synthetic-data assignment](Course3/Module2/Assignment1/assignment1.py#L723) |

## PyTorch and ecosystem API index

This is a compact catalog of the standard APIs demonstrated in the course. Search this document for the API name, then follow the linked lesson or landmark above.

### Tensors, autograd, and devices

- Creation/conversion: `torch.tensor`, `torch.zeros`, `torch.ones`, `torch.arange`, `torch.linspace`, `torch.rand`, `torch.randn`, `torch.from_numpy`, `.numpy`, `.clone`, `.detach`.
- Shape/data operations: `.shape`, `.size`, `.ndim`, `.reshape`, `.view`, `.flatten`, `.squeeze`, `.unsqueeze`, `.transpose`, `.permute`, `torch.cat`, `torch.stack`.
- Selection/math: slicing, boolean masks, broadcasting, `torch.where`, `torch.mean`, `torch.std`, `torch.sum`, `torch.max`, `torch.argmax`, `torch.softmax`, `torch.topk`, type casts and `.item`.
- Gradients/devices: `requires_grad`, `.backward`, `.grad`, `torch.no_grad`, `torch.inference_mode`, `.to`, `torch.device`, `torch.cuda.is_available`, CUDA memory APIs.
- Main references: [tensor fundamentals](Course1/Module1/Lab3/lab3.py), [first training loop](Course1/Module1/Lab1/lab1.py), [performance measurement](Course2/Module4/Lab3/lab3.py#L227).

### Neural networks, losses, and optimization

- Modules/layers: `nn.Module`, `nn.Sequential`, `nn.Linear`, `nn.Conv2d`, `nn.BatchNorm2d`, `nn.ReLU`, `nn.MaxPool2d`, `nn.AdaptiveAvgPool2d`, `nn.Dropout`, `nn.Flatten`, `nn.Embedding`, `nn.EmbeddingBag`, `nn.MultiheadAttention`, Transformer encoder/decoder layers.
- Losses: `nn.MSELoss`, `nn.CrossEntropyLoss`, contrastive loss, triplet loss, class-weighted loss.
- Optimizers/schedulers: `optim.SGD`, `optim.Adam`, `optim.AdamW`, `zero_grad`, `step`, `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`.
- State/inspection: `state_dict`, `load_state_dict`, `named_modules`, `named_parameters`, parameter counting, forward/backward hooks, `torch.save`, `torch.load`.
- Main references: [modular CNN](Course1/Module4/Ass1/c1_m4_ass1.py#L148), [schedulers](Course2/Module1/Lab2/lab2.py#L244), [hooks and Grad-CAM](Course3/Module2/Lab2/lab2.py#L265).

### Data pipelines and TorchVision

- Data APIs: `torch.utils.data.Dataset`, `DataLoader`, `Subset`, `random_split`, custom collate functions and samplers.
- Loader performance: `batch_size`, `shuffle`, `num_workers`, `pin_memory`, `prefetch_factor`, persistent workers.
- Transforms: `Compose`, `ToTensor`, `PILToTensor`, `ToPILImage`, `Resize`, `CenterCrop`, `RandomCrop`, `RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, `Normalize`.
- Datasets/utilities: `MNIST`, `EMNIST`, `FashionMNIST`, `CIFAR10`, `CIFAR100`, `SVHN`, `FakeData`, `OxfordIIITPet`, `ImageFolder`, `make_grid`, `save_image`, image decoding.
- Models/weights: ResNet, MobileNetV3, DenseNet, SqueezeNet, DeepLabV3 and Faster R-CNN constructors; pretrained weights, `.transforms()` and weights metadata.
- Main references: [custom image dataset](Course1/Module3/Ass1/ass1.py#L30), [TorchVision transforms](Course2/Module2/Lab1/lab1.py), [dataset catalog](Course2/Module2/Lab2/lab2.py), [pretrained tasks](Course2/Module2/Lab3/lab3.py).

### Metrics, tuning, training frameworks, and deployment

- Metrics: torchmetrics `Accuracy`, `F1Score`, `Precision`, `Recall`; macro averaging; confusion matrices; per-class metrics.
- Optuna: `create_study`, `study.optimize`, `Trial.suggest_*`, fixed trials, TPE/Grid samplers, optimization history, parameter importance and parallel-coordinate plots.
- Lightning: `LightningModule`, `LightningDataModule`, `Trainer`, callbacks, early stopping, precision and gradient accumulation, `PyTorchProfiler`.
- Hugging Face: `AutoTokenizer`, `BertTokenizerFast`, `AutoModelForSequenceClassification`, `DataCollatorWithPadding`, model freezing and fine-tuning.
- Diffusers: `StableDiffusionPipeline`, `DDPMPipeline`, seeded `torch.Generator`, callbacks, guidance scale and negative prompts.
- Tracking/deployment: MLflow runs/parameters/metrics/artifacts/models; `torch.onnx.export`; ONNX checker/runtime; ONNX-TF; TensorFlow SavedModel.
- Compression: `torch.nn.utils.prune` random/L1/structured/global pruning and `prune.remove`; dynamic/static quantization, observers, calibration, fusion, QAT `prepare`/`convert`.
- Main references: [metrics](Course2/Module1/Lab1/lab1.py#L251), [Optuna](Course2/Module1/Lab3/lab3.py#L137), [Lightning](Course2/Module4/Assignment1/assignment1.py#L131), [MLflow](Course3/Module4/Lab1/lab1.py#L361), [deployment](Course3/Module4/Lab2/lab2.py), [compression assignment](Course3/Module4/Assignment1/assignment1.py#L262).

## Notebook and Python-copy notes

- **Known incomplete duplicate:** [Course 3 Module 2 Assignment](Course3/Module2/Assignment1/assignment1.py#L809) retains an incomplete first `generate_sd_image` definition (`generator = None` and `image = None(...)`). A complete definition later in the same file overrides it at [line 866](Course3/Module2/Assignment1/assignment1.py#L866), so normal execution uses the complete version, but the dead placeholder should eventually be removed.
- **Intentional split:** Course 3 Module 1 Lab 1 is represented by [lab1a.py](Course3/Module1/Lab1/lab1a.py), [lab1b.py](Course3/Module1/Lab1/lab1b.py), and [training_functions.py](Course3/Module1/Lab1/training_functions.py). Together they correspond to the notebook.
- **Intentional split:** Course 2 Module 1 Lab 4 keeps models in [model_architectures.py](Course2/Module1/Lab4/model_architectures.py) and experiments in [lab4.py](Course2/Module1/Lab4/lab4.py).
- **No notebook counterpart:** [Course 1 Module 2 Lab 1](Course1/Module2/Lab1/lab1.py) has no neighboring `.ipynb` file.
- **Minor script-only helpers:** Course 2 Module 3 Labs 3 and 4 add a `decompress_gzip` helper that is not part of the corresponding notebook's main learning content.
- No empty primary lab or assignment scripts were found. A few procedural exports have no custom function/class definitions, but their learning workflows are present.

## Maintenance convention

When adding or revising course material:

1. Add one row to the relevant module table.
2. Add the material to one or more entries under **Find by task**.
3. Link to the most reusable function/class, not only the top of the script.
4. Update the dataset, model, or API indexes only when a new concept is introduced.