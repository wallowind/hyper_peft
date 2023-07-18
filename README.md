# hyper_peft
This is a fast and not very well thought out implementation of the improved LoRa (Low Rank Adaptation) technique from the article **"HyperDreamBooth: HyperNetworks for Fast Personalisation of Text-to-Image Models"**.

It is based on MonkeyPatched Peft (https://github.com/huggingface/peft) and "just works" (not thoroughly tested). I'm not sure about the correctness of the orthogonalisation implemented - the standard `torch.nn.init.orthogonal_` function was used, which probably doesn't give the correct `aux layers` (from the article: "aux layers are randomly initialised with orthogonal vectors").

Main file with implementation is `hyper_peft.py`. Two other files — `example_train.py` and `example_inference.py` shows how to use it in practice.
###### NOTE: 
The latest version of Peft (0.4.0 as of today) does not work with my MonkeyPatch and there is no easy way to fix this. There are no plans to write an another version for their new LoRa implementation.
