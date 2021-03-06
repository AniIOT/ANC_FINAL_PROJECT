---
title: ANC_FINAL_PROJECT
---

## Contributors:

| Names | Aniruddha Anand Damle | Prakriti Biswas | Aditya Shrikant Kaduskar |
|-------|-----------------------|-----------------|--------------------------|
| Email | adamle5@asu.edu       | pbiswa12@asu.edu| akaduska@asu.edu         |
| ASU ID| 1222585013            | 1222851266      | 1222545896               |

## Introduction:
This report is a discussion of our class project where we replicate the research paper 'Deep Learning-Based Gait Recognition
Using Smartphones in the Wild' by Qin Zhou et al., 2020[1]. We attempt to replicate the models used in the aforementioned paper for the purpose of gait identification and authentication. As a first step of the purpose, we extract gait features from the accelerometer and gyroscope, which are readily available in smartphones. We used an application called 'AndroSensor' from the Google Playstore. This application generates a text file or sends an email to the user with the phone's accelerometer and gyroscope data. Our classmates utilized the aforementioned software and gave us with their gait statistics. We then applied this additional dataset to the codes provided by the original paper's authors. We thoroughly examined the research and applied the methods given by the original study's authors for gait identification utilizing cellphones. We cover the implementation of the various methodologies, the experimental findings, and any issues encountered throughout the course of this project in this report.

<img src="https://media.giphy.com/media/Ejl1GWbk00HtkxH40h/giphy.gif" width="400"/>

### Final Code files
* [AndroSensor](https://play.google.com/store/apps/details?id=com.fivasim.androsensor&hl=en_US&gl=US): Use this app to collect your own data. Click on the link to learn more.
* [Data Preprocessing](/04_Software/04_Release/Gait-Recognition/00_data_preprocessing/data_preprocessing.md): Use these files first to preprocess the data if using your own dataset. Click on the link to learn more.
* [Data Extraction](/04_Software/04_Release/Gait-Recognition/01_gait_extraction/gait_extraction.md): Use these files to classify walking and non-walking data. Click on the link to learn more.
* [Identification](/04_Software/04_Release/Gait-Recognition/02_identification/identification.md): Use these files to do gait identification. Click on the link to learn more.
* [Authentication](/04_Software/04_Release/Gait-Recognition/03_authentication/authentication.md): Use these files to do gait authentication. Click on the link to learn more.

### Author's GitHub
* [Gait-Recognition-Using-Smartphones](https://github.com/qinnzou/Gait-Recognition-Using-Smartphones)

### Dataset
* Datasets can be found [here](https://onedrive.live.com/?authkey=%21APJZLtdpd%5FJQ1Ck&id=8B12BDBA699C6D2B%219501&cid=8B12BDBA699C6D2B).

### References
[1] Q. Zou, Y. Wang, Q. Wang, Y. Zhao, and Q. Li, ???Deep Learning-Based Gait Recognition Using Smartphones in the Wild,??? IEEE Transactions on Information Forensics and Security, vol. 15. pp. 3197???3212, 2020 [Online]. Available: http://dx.doi.org/10.1109/tifs.2020.2985628
