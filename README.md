<div align="center">

<h3> Guided Slot Attention for Unsupervised Video Object Segmentation
 </h3> 
 <br/>
  <a href='https://arxiv.org/abs/2303.08314'><img src='https://img.shields.io/badge/ArXiv-2303.08314-red' /></a> 
  <br/>
  <br/>
<div>
    <a href='https://hydragon.co.kr' target='_blank'>Minhyeok Lee <sup> 1</sup> </a>&emsp;
    <a href='https://suhwan-cho.github.io' target='_blank'>Suhwan Cho <sup> 1</sup></a>&emsp;
    <a href='https://dogyoonlee.github.io' target='_blank'>Dogyoon Lee <sup> 1</sup></a>&emsp;
    <a target='_blank'>Chaewon Park <sup> 1</sup></a>&emsp;
    <a href='https://jho-yonsei.github.io' target='_blank'>Jungho Lee <sup> 1</sup></a>&emsp;
    <a target='_blank'>Sangyoun Lee <sup>1,2</sup></a>&emsp;
</div>
<br>
<div>
                      <sup>1</sup> Yonsei University &nbsp;&nbsp;&nbsp;
                      <sup>2</sup> Korea Institute of Science and Technology (KIST) &nbsp;
</div>
<br>
<i><strong><a href='https://cvpr.thecvf.com' target='_blank'>CVPR 2024</a></strong></i>
<br>
<br>
</div>

## Abstract
Unsupervised video object segmentation aims to segment the most prominent object in a video sequence. However, the existence of complex backgrounds and multiple foreground objects make this task challenging. To address this issue, we propose a guided slot attention network to reinforce spatial structural information and obtain better foreground--background separation. The foreground and background slots, which are initialized with query guidance, are iteratively refined based on interactions with template information. Furthermore, to improve slot--template interaction and effectively fuse global and local features in the target and reference frames, K-nearest neighbors filtering and a feature aggregation transformer are introduced. The proposed model achieves state-of-the-art performance on two popular datasets. Additionally, we demonstrate the robustness of the proposed model in challenging scenes through various comparative experiments.
