## PyTorch Snippets

- [pack_padded_sequence](https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html)
  
- PyTorch Seed 고정
  ```python3
  import random
  import torch
  import numpy as np
  
  def fix_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True # this can slow down speed
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
  ```

- Number of parameters
  ```python3
  num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
  ```
  
- Print formatted time
  ```python3
  import time
  def check_time(start_time: float) -> str:
    sec = time.time() - start_time
    times = str(datetime.timedelta(seconds=sec)).split(".")
    return times[0]
  # start_time = time.time()
  # total_time = check_time(start_time)
  ```
  
- Sync checkpoint parameter name 
  ```python3
  # original saved file with DataParallel
  state_dict = torch.load('myfile.pth.tar')
  # create new OrderedDict that does not contain `module.`
  from collections import OrderedDict
  new_state_dict = OrderedDict()
  for k, v in state_dict.items():
      name = k[7:] # remove `module.`
      new_state_dict[name] = v
  # load params
  model.load_state_dict(new_state_dict)
  ```
