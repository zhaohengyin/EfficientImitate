# [NeurIPS 2022] EfficientImitate
This is the codebase of our paper ''Planning for Sample Efficient Imitation Learning'' at the NeurIPS 2022.

## Preparation
**Step 1: Preparing python packages.**  This project is dependent on the following packages ``ray``, ``torch``, ``dmc2gym``, ``opencv-python``, and ``kornia``. We can use ``pip`` to install them.

**Step 2: Compiling cpp source.**  After installing these dependencies, we need to compile the C++ source of the MCTS as follows.
```
cd mcts_tree_sample
bash make.sh
```

**Step 3: Download the data.** Finally, we need to download the demonstration data at (TBD), and put them into the ``./data`` folder.

## Launch Training
We put the launch scripts at the ``./scripts`` folder. For example, you can launch the training of walker by
```
bash ./scripts/walker_state.sh
```

## Cite this work
If you find this work useful and would like to cite it in youre research:
```
@inproceedings{efficientimitate,
  title={Planning for Sample Efficient Imitation Learning},
  author={Yin, Zhao-Heng and Ye, Weirui and Chen, Qifeng and Gao, Yang},
  booktitle={Neural Information Processing Systems},
  year={2022}
}

```
## License
MIT License
