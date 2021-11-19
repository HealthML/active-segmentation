The BraTS Dataset
============

The BraTS 2018 dataset was published in the course of the annual MultimodalBrainTumorSegmentation Challenge (BraTS)
held since 2012. It is composed of 3T multimodal MRI scans from patients affected by glioblastoma or lower grade glioma,
as well as corresponding ground truth labels provided by expert board-certified neuroradiologists.

General Information
**********

Contributors from 19 different institutions from the U.S., Europe, and Asia provided data acquired by several MRI
scanners and according to different clinical protocols. All MRI scans were taken from patients who received a
pathologically confirmed diagnosis and have been subjected to common pre-processing steps, including skull-stripping
and re-sampling to a shared resolution of 240×240×155.
According to the protocol, a total of five different structures were annotated, namely “healthy spots”, “edema”,
“enhancing core”, “necrotic (or fluid-filled) core”, “non-enhancing(solid) core”.
To better represent the clinical application tasks, these structures were grouped into three mutually inclusive
categories:

1. the “whole” tumor region, including “edema”, “enhancing core”, “necrotic (or fluid-filled)core”, “non-enhancing (solid) core”

2. the “core” tumor region, including “enhancing core”, “necrotic (or fluid-filled) core”, “non-enhancing (solid) core”

3. the “active” tumor region, including “enhancing core”

Sample Size and Data Split
***********

The BraTS 2018 dataset consists of a total of 471 scans. Of these, 405 scans are labeled, and 66 are unlabeled.
By default the 405 labeled scans were split into a training dataset with 285 scans and a validation dataset with 120 scans.
To have a proper validation on a held out dataset, we decided to further split the dataset into 245 scans for training,
80 scans for validation and 80 scans as holdout set. In doing so we established a 60/20/20 split for our experiments.

Details
***********

To setup the holdout dataset following scans were moved:

1. From training dataset to holdout dataset:

* 40 files in total were moved: 24 scans from TCIA, 12 scans from CBICA, 4 scans from 2013 randomly selected

List of scans moved from training data:
::
    [
        'Brats18_TCIA01_231_1',
        'Brats18_TCIA02_309_1',
        'Brats18_TCIA03_133_1',
        'Brats18_TCIA10_282_1',
        'Brats18_TCIA02_300_1',
        'Brats18_TCIA06_603_1',
        'Brats18_TCIA02_377_1',
        'Brats18_TCIA10_449_1',
        'Brats18_TCIA01_221_1',
        'Brats18_TCIA02_370_1',
        'Brats18_TCIA08_162_1',
        'Brats18_TCIA08_205_1',
        'Brats18_TCIA10_629_1',
        'Brats18_TCIA02_473_1',
        'Brats18_TCIA02_198_1',
        'Brats18_TCIA08_167_1',
        'Brats18_TCIA02_314_1',
        'Brats18_TCIA10_387_1',
        'Brats18_TCIA09_255_1',
        'Brats18_TCIA09_620_1',
        'Brats18_TCIA06_211_1',
        'Brats18_TCIA01_429_1',
        'Brats18_TCIA02_300_1',
        'Brats18_TCIA09_402_1',
        'Brats18_CBICA_AYU_1',
        'Brats18_CBICA_ABB_1',
        'Brats18_CBICA_AQQ_1',
        'Brats18_CBICA_AQJ_1',
        'Brats18_CBICA_AUQ_1',
        'Brats18_CBICA_ARW_1',
        'Brats18_CBICA_AYA_1',
        'Brats18_CBICA_ABM_1',
        'Brats18_CBICA_AQG_1',
        'Brats18_CBICA_ABN_1',
        'Brats18_CBICA_AQY_1',
        'Brats18_CBICA_ALX_1',
        'Brats18_2013_9_1',
        'Brats18_2013_26_1',
        'Brats18_2013_19_1',
        'Brats18_2013_16_1',
    ]

2. From validation dataset to holdout dataset:

* 40 files in total were moved: 26 scans from TCIA, 10 scans from CBICA, 4 scans from 2013 randomly selected

List of scans moved from validation data:

::
    [
        'Brats17_TCIA_640_1',
        'Brats17_TCIA_242_1',
        'Brats17_TCIA_321_1',
        'Brats17_TCIA_186_1',
        'Brats17_TCIA_430_1',
        'Brats17_TCIA_151_1',
        'Brats17_TCIA_420_1',
        'Brats17_TCIA_202_1',
        'Brats17_TCIA_387_1',
        'Brats17_TCIA_298_1',
        'Brats17_TCIA_449_1',
        'Brats17_TCIA_149_1',
        'Brats17_TCIA_152_1',
        'Brats17_TCIA_498_1',
        'Brats17_TCIA_266_1',
        'Brats17_TCIA_276_1',
        'Brats17_TCIA_624_1',
        'Brats17_TCIA_618_1',
        'Brats17_TCIA_175_1',
        'Brats17_TCIA_101_1',
        'Brats17_TCIA_141_1',
        'Brats17_TCIA_343_1',
        'Brats17_TCIA_621_1',
        'Brats17_TCIA_607_1',
        'Brats17_TCIA_469_1',
        'Brats17_TCIA_282_1',
        'Brats17_CBICA_ASY_1',
        'Brats17_CBICA_AWI_1',
        'Brats17_CBICA_ASE_1',
        'Brats17_CBICA_APR_1',
        'Brats17_CBICA_BHB_1',
        'Brats17_CBICA_AQG_1',
        'Brats17_TCIA_639_1',
        'Brats17_CBICA_AZD_1',
        'Brats17_CBICA_ASA_1',
        'Brats17_CBICA_AQV_1',
        'Brats17_2013_28_1',
        'Brats17_2013_3_1',
        'Brats17_2013_24_1',
        'Brats17_2013_5_1',
    ]
