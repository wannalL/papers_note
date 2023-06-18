## Survey

A Survey of Diffusion Models in Natural Language Processing : https://arxiv.org/abs/2305.14671

> This survey paper provides a comprehensive review of the use of diffusion models in natural language processing (NLP). Diffusion models are a class of mathematical models that aim to capture the diffusion of information or signals across a network or manifold. In NLP, diffusion models have been used in a variety of applications, such as natural language generation, sentiment analysis, topic modeling, and machine translation. This paper discusses the different formulations of diffusion models used in NLP, their strengths and limitations, and their applications. We also perform a thorough comparison between diffusion models and alternative generative models, specifically highlighting the autoregressive (AR) models, while also examining how diverse architectures incorporate the Transformer in conjunction with diffusion models. Compared to AR models, diffusion models have significant advantages for parallel generation, text interpolation, token-level controls such as syntactic structures and semantic contents, and robustness. Exploring further permutations of integrating Transformers into diffusion models would be a valuable pursuit. Also, the development of multimodal diffusion models and large-scale diffusion language models with notable capabilities for few-shot learning would be important directions for the future advance of diffusion models in NLP.





A Complete Survey on Generative AI (AIGC): Is ChatGPT from GPT-4 to GPT-5 All You Need? : https://arxiv.org/abs/2303.11717

> As ChatGPT goes viral, generative AI (AIGC, a.k.a AI-generated content) has made headlines everywhere because of its ability to analyze and create text, images, and beyond. With such overwhelming media coverage, it is almost impossible for us to miss the opportunity to glimpse AIGC from a certain angle. In the era of AI transitioning from pure analysis to creation, it is worth noting that ChatGPT, with its most recent language model GPT-4, is just a tool out of numerous AIGC tasks. Impressed by the capability of the ChatGPT, many people are wondering about its limits: can GPT-5 (or other future GPT variants) help ChatGPT unify all AIGC tasks for diversified content creation? Toward answering this question, a comprehensive review of existing AIGC tasks is needed. As such, our work comes to fill this gap promptly by offering a first look at AIGC, ranging from its techniques to applications. Modern generative AI relies on various technical foundations, ranging from model architecture and self-supervised pretraining to generative modeling methods (like GAN and diffusion models). After introducing the fundamental techniques, this work focuses on the technological development of various AIGC tasks based on their output type, including text, images, videos, 3D content, etc., which depicts the full potential of ChatGPT's future. Moreover, we summarize their significant applications in some mainstream industries, such as education and creativity content. Finally, we discuss the challenges currently faced and present an outlook on how generative AI might evolve in the near future.



## AIGC

### CV

ViCo : https://arxiv.org/abs/2306.00971

> ViCo : Detail-Preserving Visual Condition for Personalized Text-to-Image Generation
>
> Personalized text-to-image generation using diffusion models has recently been proposed and attracted lots of attention. Given a handful of images containing a novel concept (e.g., a unique toy), we aim to tune the generative model to capture fine visual details of the novel concept and generate photorealistic images following a text condition. We present a plug-in method, named ViCo, for fast and lightweight personalized generation. Specifically, we propose an image attention module to condition the diffusion process on the patch-wise visual semantics. We introduce an attention-based object mask that comes almost at no cost from the attention module. In addition, we design a simple regularization based on the intrinsic properties of text-image attention maps to alleviate the common overfitting degradation. Unlike many existing models, our method does not finetune any parameters of the original diffusion model. This allows more flexible and transferable model deployment. With only light parameter training (~6% of the diffusion U-Net), our method achieves comparable or even better performance than all state-of-the-art models both qualitatively and quantitatively.


IJEPA : https://arxiv.org/abs/2301.08243

> Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
>
> This paper demonstrates an approach for learning highly semantic image representations without relying on hand-crafted data-augmentations. We introduce the Image-based Joint-Embedding Predictive Architecture (I-JEPA), a non-generative approach for self-supervised learning from images. The idea behind I-JEPA is simple: from a single context block, predict the representations of various target blocks in the same image. A core design choice to guide I-JEPA towards producing semantic representations is the masking strategy; specifically, it is crucial to (a) sample target blocks with sufficiently large scale (semantic), and to (b) use a sufficiently informative (spatially distributed) context block. Empirically, when combined with Vision Transformers, we find I-JEPA to be highly scalable. For instance, we train a ViT-Huge/14 on ImageNet using 16 A100 GPUs in under 72 hours to achieve strong downstream performance across a wide range of tasks, from linear classification to object counting and depth prediction.


Collaborative Diffusion for Multi-Modal Face Generation and Editing : https://arxiv.org/abs/2304.10530

> Diffusion models arise as a powerful generative tool recently. Despite the great progress, existing diffusion models mainly focus on uni-modal control, i.e., the diffusion process is driven by only one modality of condition. To further unleash the users' creativity, it is desirable for the model to be controllable by multiple modalities simultaneously, e.g., generating and editing faces by describing the age (text-driven) while drawing the face shape (mask-driven). In this work, we present Collaborative Diffusion, where pre-trained uni-modal diffusion models collaborate to achieve multi-modal face generation and editing without re-training. Our key insight is that diffusion models driven by different modalities are inherently complementary regarding the latent denoising steps, where bilateral connections can be established upon. Specifically, we propose dynamic diffuser, a meta-network that adaptively hallucinates multi-modal denoising steps by predicting the spatial-temporal influence functions for each pre-trained uni-modal model. Collaborative Diffusion not only collaborates generation capabilities from uni-modal diffusion models, but also integrates multiple uni-modal manipulations to perform multi-modal editing. Extensive qualitative and quantitative experiments demonstrate the superiority of our framework in both image quality and condition consistency

DreamFusion: Text-to-3D using 2D Diffusion : https://arxiv.org/abs/2209.14988

> Recent breakthroughs in text-to-image synthesis have been driven by diffusion models trained on billions of image-text pairs. Adapting this approach to 3D synthesis would require large-scale datasets of labeled 3D data and efficient architectures for denoising 3D data, neither of which currently exist. In this work, we circumvent these limitations by using a pretrained 2D text-to-image diffusion model to perform text-to-3D synthesis. We introduce a loss based on probability density distillation that enables the use of a 2D diffusion model as a prior for optimization of a parametric image generator. Using this loss in a DeepDream-like procedure, we optimize a randomly-initialized 3D model (a Neural Radiance Field, or NeRF) via gradient descent such that its 2D renderings from random angles achieve a low loss. The resulting 3D model of the given text can be viewed from any angle, relit by arbitrary illumination, or composited into any 3D environment. Our approach requires no 3D training data and no modifications to the image diffusion model, demonstrating the effectiveness of pretrained image diffusion models as priors.



PaLI: A Jointly-Scaled Multilingual Language-Image Model

> Effective scaling and a flexible task interface enable large language models to excel at many tasks. We present PaLI (Pathways Language and Image model), a model that extends this approach to the joint modeling of language and vision. PaLI generates text based on visual and textual inputs, and with this interface performs many vision, language, and multimodal tasks, in many languages. To train PaLI, we make use of large pre-trained encoder-decoder language models and Vision Transformers (ViTs). This allows us to capitalize on their existing capabilities and leverage the substantial cost of training them. We find that joint scaling of the vision and language components is important. Since existing Transformers for language are much larger than their vision counterparts, we train a large, 4-billion parameter ViT (ViT-e) to quantify the benefits from even larger-capacity vision models. To train PaLI, we create a large multilingual mix of pretraining tasks, based on a new image-text training set containing 10B images and texts in over 100 languages. PaLI achieves state-of-the-art in multiple vision and language tasks (such as captioning, visual question-answering, scene-text understanding), while retaining a simple, modular, and scalable design.



Token Merging: Your ViT But Faster: https://arxiv.org/abs/2210.09461

> We introduce Token Merging (ToMe), a simple method to increase the throughput of existing ViT models without needing to train. ToMe gradually combines similar tokens in a transformer using a general and light-weight matching algorithm that is as fast as pruning while being more accurate. Off-the-shelf, ToMe can 2x the throughput of state-of-the-art ViT-L @ 512 and ViT-H @ 518 models on images and 2.2x the throughput of ViT-L on video with only a 0.2-0.3% accuracy drop in each case. ToMe can also easily be applied during training, improving in practice training speed up to 2x for MAE fine-tuning on video. Training with ToMe further minimizes accuracy drop, leading to 2x the throughput of ViT-B on audio for only a 0.4% mAP drop. Qualitatively, we find that ToMe merges object parts into one token, even over multiple frames of video. Overall, ToMe's accuracy and speed are competitive with state-of-the-art on images, video, and audio.


### AS

MusicLM: Generating Music From Text : https://arxiv.org/abs/2301.11325

> We introduce MusicLM, a model generating high-fidelity music from text descriptions such as "a calming violin melody backed by a distorted guitar riff". MusicLM casts the process of conditional music generation as a hierarchical sequence-to-sequence modeling task, and it generates music at 24 kHz that remains consistent over several minutes. Our experiments show that MusicLM outperforms previous systems both in audio quality and adherence to the text description. Moreover, we demonstrate that MusicLM can be conditioned on both text and a melody in that it can transform whistled and hummed melodies according to the style described in a text caption. To support future research, we publicly release MusicCaps, a dataset composed of 5.5k music-text pairs, with rich text descriptions provided by human experts.



## ELSE

DIFFormer: Scalable (Graph) Transformers Induced by Energy Constrained Diffusion : https://arxiv.org/abs/2301.09474

> Real-world data generation often involves complex inter-dependencies among instances, violating the IID-data hypothesis of standard learning paradigms and posing a challenge for uncovering the geometric structures for learning desired instance representations. To this end, we introduce an energy constrained diffusion model which encodes a batch of instances from a dataset into evolutionary states that progressively incorporate other instances' information by their interactions. The diffusion process is constrained by descent criteria w.r.t.~a principled energy function that characterizes the global consistency of instance representations over latent structures. We provide rigorous theory that implies closed-form optimal estimates for the pairwise diffusion strength among arbitrary instance pairs, which gives rise to a new class of neural encoders, dubbed as DIFFormer (diffusion-based Transformers), with two instantiations: a simple version with linear complexity for prohibitive instance numbers, and an advanced version for learning complex structures. Experiments highlight the wide applicability of our model as a general-purpose encoder backbone with superior performance in various tasks, such as node classification on large graphs, semi-supervised image/text classification, and spatial-temporal dynamics prediction.



On the duality between contrastive and non-contrastive self-supervised learning : https://arxiv.org/abs/2206.02574

> Recent approaches in self-supervised learning of image representations can be categorized into different families of methods and, in particular, can be divided into contrastive and non-contrastive approaches. While differences between the two families have been thoroughly discussed to motivate new approaches, we focus more on the theoretical similarities between them. By designing contrastive and covariance based non-contrastive criteria that can be related algebraically and shown to be equivalent under limited assumptions, we show how close those families can be. We further study popular methods and introduce variations of them, allowing us to relate this theoretical result to current practices and show the influence (or lack thereof) of design choices on downstream performance. Motivated by our equivalence result, we investigate the low performance of SimCLR and show how it can match VICReg's with careful hyperparameter tuning, improving significantly over known baselines. We also challenge the popular assumptions that contrastive and non-contrastive methods, respectively, need large batch sizes and output dimensions. Our theoretical and quantitative results suggest that the numerical gaps between contrastive and non-contrastive methods in certain regimes can be closed given better network design choices and hyperparameter tuning. The evidence shows that unifying different SOTA methods is an important direction to build a better understanding of self-supervised learning.
