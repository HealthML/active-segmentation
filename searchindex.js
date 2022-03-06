Search.setIndex({docnames:["active_learning","active_segmentation","brats","datasets","functional","index","inferencing","main","metric_tracking","models","modules","query_strategies","run_experiments"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["active_learning.rst","active_segmentation.rst","brats.rst","datasets.rst","functional.rst","index.rst","inferencing.rst","main.rst","metric_tracking.rst","models.rst","modules.rst","query_strategies.rst","run_experiments.rst"],objects:{"":{active_learning:[0,0,0,"-"],inferencing:[6,0,0,"-"],main:[7,0,0,"-"],run_experiments:[12,0,0,"-"]},"active_learning.ActiveLearningPipeline":{remove_wandb_cache:[0,2,1,""],run:[0,2,1,""],setup_trainer:[0,2,1,""]},"datasets.brats_data_module":{BraTSDataModule:[3,1,1,""]},"datasets.brats_data_module.BraTSDataModule":{discover_paths:[3,2,1,""],id_to_class_names:[3,2,1,""],label_items:[3,2,1,""],multi_label:[3,2,1,""],train_dataloader:[3,2,1,""]},"datasets.data_module":{ActiveLearningDataModule:[3,1,1,""]},"datasets.data_module.ActiveLearningDataModule":{data_channels:[3,2,1,""],id_to_class_names:[3,2,1,""],label_items:[3,2,1,""],multi_label:[3,2,1,""],num_classes:[3,2,1,""],setup:[3,2,1,""],test_dataloader:[3,2,1,""],test_set_size:[3,2,1,""],train_dataloader:[3,2,1,""],training_set_size:[3,2,1,""],unlabeled_dataloader:[3,2,1,""],unlabeled_set_size:[3,2,1,""],val_dataloader:[3,2,1,""],validation_set_size:[3,2,1,""]},"functional.losses":{AbstractDiceLoss:[4,1,1,""],CrossEntropyDiceLoss:[4,1,1,""],CrossEntropyLoss:[4,1,1,""],DiceLoss:[4,1,1,""],FalsePositiveDiceLoss:[4,1,1,""],FalsePositiveLoss:[4,1,1,""],FocalLoss:[4,1,1,""],GeneralizedDiceLoss:[4,1,1,""],SegmentationLoss:[4,1,1,""]},"functional.losses.AbstractDiceLoss":{forward:[4,2,1,""],get_dice_loss_module:[4,2,1,""],training:[4,3,1,""]},"functional.losses.CrossEntropyDiceLoss":{forward:[4,2,1,""],training:[4,3,1,""]},"functional.losses.CrossEntropyLoss":{forward:[4,2,1,""],training:[4,3,1,""]},"functional.losses.DiceLoss":{forward:[4,2,1,""],get_dice_loss_module:[4,2,1,""],training:[4,3,1,""]},"functional.losses.FalsePositiveDiceLoss":{forward:[4,2,1,""],training:[4,3,1,""]},"functional.losses.FalsePositiveLoss":{forward:[4,2,1,""],training:[4,3,1,""]},"functional.losses.FocalLoss":{training:[4,3,1,""]},"functional.losses.GeneralizedDiceLoss":{forward:[4,2,1,""],get_dice_loss_module:[4,2,1,""],training:[4,3,1,""]},"functional.losses.SegmentationLoss":{training:[4,3,1,""]},"functional.metrics":{DiceScore:[4,1,1,""],HausdorffDistance:[4,1,1,""],SegmentationMetric:[4,1,1,""],Sensitivity:[4,1,1,""],Specificity:[4,1,1,""],dice_score:[4,4,1,""],hausdorff_distance:[4,4,1,""],sensitivity:[4,4,1,""],single_class_hausdorff_distance:[4,4,1,""],specificity:[4,4,1,""]},"functional.metrics.DiceScore":{compute:[4,2,1,""],update:[4,2,1,""]},"functional.metrics.HausdorffDistance":{compute:[4,2,1,""],reset:[4,2,1,""],update:[4,2,1,""]},"functional.metrics.Sensitivity":{compute:[4,2,1,""],update:[4,2,1,""]},"functional.metrics.Specificity":{compute:[4,2,1,""],update:[4,2,1,""]},"inferencing.Inferencer":{inference:[6,2,1,""],inference_image:[6,2,1,""],inference_scan:[6,2,1,""]},"metric_tracking.combined_per_epoch_metric":{CombinedPerEpochMetric:[8,1,1,""]},"metric_tracking.combined_per_epoch_metric.CombinedPerEpochMetric":{compute:[8,2,1,""],get_metric_names:[8,2,1,""],reset:[8,2,1,""],update:[8,2,1,""]},"metric_tracking.combined_per_image_metric":{CombinedPerImageMetric:[8,1,1,""]},"metric_tracking.combined_per_image_metric.CombinedPerImageMetric":{compute:[8,2,1,""],reset:[8,2,1,""],update:[8,2,1,""]},"models.pytorch_fcn_resnet50":{PytorchFCNResnet50:[9,1,1,""]},"models.pytorch_fcn_resnet50.PytorchFCNResnet50":{eval:[9,2,1,""],forward:[9,2,1,""],input_dimensionality:[9,2,1,""],parameters:[9,2,1,""],reset_parameters:[9,2,1,""],test_step:[9,2,1,""],train:[9,2,1,""],training:[9,3,1,""],training_step:[9,2,1,""],validation_step:[9,2,1,""]},"models.pytorch_model":{PytorchModel:[9,1,1,""]},"models.pytorch_model.PytorchModel":{configure_loss:[9,2,1,""],configure_metrics:[9,2,1,""],configure_optimizers:[9,2,1,""],current_epoch:[9,5,1,""],get_test_metrics:[9,2,1,""],get_train_metrics:[9,2,1,""],get_val_metrics:[9,2,1,""],input_dimensionality:[9,2,1,""],predict:[9,2,1,""],predict_step:[9,2,1,""],reset_parameters:[9,2,1,""],setup:[9,2,1,""],test_epoch_end:[9,2,1,""],test_step:[9,2,1,""],training:[9,3,1,""],training_epoch_end:[9,2,1,""],training_step:[9,2,1,""],validation_epoch_end:[9,2,1,""],validation_step:[9,2,1,""]},"models.pytorch_u_net":{PytorchUNet:[9,1,1,""]},"models.pytorch_u_net.PytorchUNet":{eval:[9,2,1,""],forward:[9,2,1,""],input_dimensionality:[9,2,1,""],parameters:[9,2,1,""],precision:[9,3,1,""],predict_step:[9,2,1,""],reset_parameters:[9,2,1,""],test_step:[9,2,1,""],train:[9,2,1,""],training:[9,3,1,""],training_step:[9,2,1,""],use_amp:[9,3,1,""],validation_step:[9,2,1,""]},"models.u_net":{UNet:[9,1,1,""]},"models.u_net.UNet":{forward:[9,2,1,""],training:[9,3,1,""]},"query_strategies.query_strategy":{QueryStrategy:[11,1,1,""]},"query_strategies.query_strategy.QueryStrategy":{select_items_to_label:[11,2,1,""]},active_learning:{ActiveLearningPipeline:[0,1,1,""]},datasets:{brats_data_module:[3,0,0,"-"],data_module:[3,0,0,"-"]},functional:{losses:[4,0,0,"-"],metrics:[4,0,0,"-"]},inferencing:{Inferencer:[6,1,1,""]},main:{create_data_module:[7,4,1,""],create_model:[7,4,1,""],create_query_strategy:[7,4,1,""],run_active_learning_pipeline:[7,4,1,""],run_active_learning_pipeline_from_config:[7,4,1,""]},metric_tracking:{combined_per_epoch_metric:[8,0,0,"-"],combined_per_image_metric:[8,0,0,"-"]},models:{pytorch_fcn_resnet50:[9,0,0,"-"],pytorch_model:[9,0,0,"-"],pytorch_u_net:[9,0,0,"-"],u_net:[9,0,0,"-"]},query_strategies:{query_strategy:[11,0,0,"-"]},run_experiments:{create_config_files:[12,4,1,""],create_sbatch_jobs_from_config_files:[12,4,1,""],start_sbatch_runs:[12,4,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"],"5":["py","property","Python property"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function","5":"py:property"},terms:{"0":[0,3,4,7,8,9],"0001":[7,9],"03385":9,"04597":9,"1":[0,3,4,7,9],"10":[2,4],"1193":0,"12":2,"120":2,"1249":4,"1505":9,"1512":9,"155":2,"16":7,"19":2,"1e":4,"2":[3,4,9],"20":2,"2012":2,"2013":2,"2016":4,"2017":4,"2018":2,"24":2,"240":2,"245":2,"26":2,"285":2,"2d":[3,4],"3":[3,4,9],"32":9,"3d":[3,4,8,9],"3t":2,"4":[2,7,9],"40":2,"405":2,"42":7,"471":2,"5":[4,7,8,9],"50":[7,9],"60":2,"66":2,"80":2,"95":4,"abstract":[3,4,9,11],"boolean":3,"case":[4,8],"class":[0,3,4,6,8,9,11],"default":[0,2,3,4,7,8,9,12],"do":2,"final":8,"float":[4,7,8,9],"function":[1,7,10,12],"int":[0,3,4,7,8,9],"new":[0,8],"return":[0,3,4,7,8,9,11],"static":[0,3,9],"true":[0,3,4,7,8,9,12],"while":[7,9],A:[0,3,4,6,7,8,9],As:4,By:2,For:[4,12],If:[0,4,7,8,9],In:[2,4,8],It:[2,4],Of:2,The:[0,3,4,5,6,7,8,9,12],To:2,_:8,_aggregated_:8,abc:[3,4,9,11],abl:0,about:9,abov:8,abstractdiceloss:4,accord:2,accordingli:8,acquir:2,across:[3,8],activ:[0,2,3,7,9],active_learn:[1,10],active_learning_config:7,active_learning_framework:5,active_learning_mod:[0,3],activelearningdatamodul:[0,3,7],activelearningpipelin:0,adam:9,adapt:[4,9],addit:[0,9,11],addition:[4,8],addtion:4,affect:2,after:0,aggreg:[4,8],ahmad:4,ahmadi:4,algorithm:[0,7,9],all:[0,2,3,4,7,8,9],all_image_loc:4,allow:12,also:8,amount:[3,6],an:[0,3,4,7,9,11,12],analog:4,ani:[3,7,9],annot:[2,3],annual:2,anymor:[0,7],appli:[4,9],applic:2,ar:[0,2,4,7,8,9,12],arbitrari:4,architectur:[0,7,9],arg:[3,4],arrai:11,arxiv:9,asia:2,assum:8,avail:[8,9],averag:4,avoid:4,b:7,background:[4,8,9],base:[0,3,4,6,7,8,9,11],batch:[3,4,8,9],batch_idx:9,batch_siz:[3,7],batch_size_unlabeled_set:3,bce:4,been:2,befor:9,begin:9,behaviour:4,best:[0,7],better:2,between:[3,4],bias:[0,7],binari:4,blob:9,block:[7,9],board:2,bool:[0,3,4,7,8,9,12],both:[4,8],brain:9,brat:[3,5,6,7],brats17_2013_24_1:2,brats17_2013_28_1:2,brats17_2013_3_1:2,brats17_2013_5_1:2,brats17_cbica_apr_1:2,brats17_cbica_aqg_1:2,brats17_cbica_aqv_1:2,brats17_cbica_asa_1:2,brats17_cbica_ase_1:2,brats17_cbica_asy_1:2,brats17_cbica_awi_1:2,brats17_cbica_azd_1:2,brats17_cbica_bhb_1:2,brats17_tcia_101_1:2,brats17_tcia_141_1:2,brats17_tcia_149_1:2,brats17_tcia_151_1:2,brats17_tcia_152_1:2,brats17_tcia_175_1:2,brats17_tcia_186_1:2,brats17_tcia_202_1:2,brats17_tcia_242_1:2,brats17_tcia_266_1:2,brats17_tcia_276_1:2,brats17_tcia_282_1:2,brats17_tcia_298_1:2,brats17_tcia_321_1:2,brats17_tcia_343_1:2,brats17_tcia_387_1:2,brats17_tcia_420_1:2,brats17_tcia_430_1:2,brats17_tcia_449_1:2,brats17_tcia_469_1:2,brats17_tcia_498_1:2,brats17_tcia_607_1:2,brats17_tcia_618_1:2,brats17_tcia_621_1:2,brats17_tcia_624_1:2,brats17_tcia_639_1:2,brats17_tcia_640_1:2,brats18_2013_16_1:2,brats18_2013_19_1:2,brats18_2013_26_1:2,brats18_2013_9_1:2,brats18_cbica_abb_1:2,brats18_cbica_abm_1:2,brats18_cbica_abn_1:2,brats18_cbica_alx_1:2,brats18_cbica_aqg_1:2,brats18_cbica_aqj_1:2,brats18_cbica_aqq_1:2,brats18_cbica_aqy_1:2,brats18_cbica_arw_1:2,brats18_cbica_auq_1:2,brats18_cbica_aya_1:2,brats18_cbica_ayu_1:2,brats18_tcia01_221_1:2,brats18_tcia01_231_1:2,brats18_tcia01_429_1:2,brats18_tcia02_198_1:2,brats18_tcia02_300_1:2,brats18_tcia02_309_1:2,brats18_tcia02_314_1:2,brats18_tcia02_370_1:2,brats18_tcia02_377_1:2,brats18_tcia02_473_1:2,brats18_tcia03_133_1:2,brats18_tcia06_211_1:2,brats18_tcia06_603_1:2,brats18_tcia08_162_1:2,brats18_tcia08_167_1:2,brats18_tcia08_205_1:2,brats18_tcia09_255_1:2,brats18_tcia09_402_1:2,brats18_tcia09_620_1:2,brats18_tcia10_282_1:2,brats18_tcia10_387_1:2,brats18_tcia10_449_1:2,brats18_tcia10_629_1:2,brats_data_modul:3,bratsdatamodul:[1,10],bratsdataset:[1,10],c:[4,8],cach:[0,7],cache_s:3,calcul:[4,8,9],call:9,can:[3,4,8,12],cardoso:4,carol:4,categori:2,cbica:2,cdot:4,certifi:2,challeng:2,channel:[3,4,8,9],checkpoint:[0,7],checkpoint_dir:[0,7],chosen:7,class_nam:8,classif:[4,9],classifi:4,cleanup:0,clear_wandb_cach:[0,7],cli:7,client:0,clinic:2,coeffici:4,com:[0,4,9],combin:[4,12],combined_per_epoch_metr:[1,10],combined_per_image_metr:[1,10],combinedperepochmetr:8,combinedperimagemetr:8,common:2,compens:0,compos:2,comput:[4,8,9],confid:[8,9],confidence_level:8,config:[7,12],config_dir:12,config_file_nam:7,config_file_path:12,configur:[3,7,9],configure_loss:9,configure_metr:9,configure_optim:9,confirm:2,consid:4,consist:2,constant:7,contain:[0,3,4,6,8,11,12],contribut:4,contributor:2,convent:3,convert:4,convert_to_one_hot:4,convolut:4,core:[2,3,9],correct:7,correspond:[2,11,12],cosineannealinglr:[0,7,9],cours:2,creat:[3,7,12],create_config_fil:12,create_data_modul:7,create_model:7,create_query_strategi:7,create_sbatch_jobs_from_config_fil:12,creation:3,criterion:9,cross:4,cross_entropi:9,cross_entropy_dic:9,crossentropydiceloss:4,crossentropyloss:4,cuda:[0,7],current:[0,3,9,11],current_epoch:9,custom:4,data:[0,3,4,5,6,7,8,9,11],data_channel:3,data_dir:[3,6,7],data_modul:[0,3,7,11],dataload:[3,9,11],dataloader_id:9,dataloader_idx:9,datamodul:[1,10],dataset:[0,1,5,6,7,8,9,10,11,12],dataset_config:[6,7,12],datset:3,decid:2,decod:[7,9],deep:4,defin:[0,3,9],delet:[0,7],denomin:4,descript:[3,8],desir:7,detail:[0,5,9],determin:[4,8,9],determinist:[0,7],deterministic_mod:[0,7],diagnosi:2,diagon:4,dice:[4,8,9],dice_scor:[4,8,9],diceloss:4,dicescor:4,dict:[3,7,8,9,11],dictionari:[7,8,9],dictonari:11,differ:[2,4,7,8],differenti:4,dim:[3,9],dimens:[4,9],dimension:[4,9],dir_path:3,directori:[0,3,6,7,12],disabl:[0,7],discov:3,discover_path:3,discuss:4,distanc:[4,8],divid:4,divis:4,dl:4,document:9,doe:[0,4],doubl:9,down:[4,9],dropout:9,dsc:4,dure:[0,7,9],dynam:[0,7,9],e:[0,6,7,8],each:[4,7,8,9,12],earli:[0,7],early_stop:[0,7],edema:2,either:[4,8,9],element:4,en:[4,9],enabl:[0,7],encod:[4,7,8,9],end:9,enhanc:2,ensur:4,entropi:4,environ:0,epoch:[0,3,7,8,9],epochs_increase_per_queri:0,epsilon:4,equal:[4,9],establish:2,euclidean:4,europ:2,eval:9,evalu:[4,9],everi:0,exampl:[4,7],exclud:[4,8,9],execut:[0,6,7,9],expect:8,experi:[0,2,7],experiment_nam:7,experiment_tag:7,expert:2,factor:4,fals:[0,3,4,7,8,9],falsepositivediceloss:4,falsepositiveloss:4,far:4,fausto:4,fcn_resnet50:9,featur:9,file:[0,2,3,7,12],fill:2,finish:[0,7,12],first:[8,9],fit:[0,9],five:2,flag:[3,7],flair:3,flexibl:9,fluid:2,fn:4,focal:[4,9],focalloss:4,folder:12,follow:2,form:8,formul:4,forward:[4,9],fp:[4,9],fp_dice:9,frac:4,framework:[4,9],from:[2,3,4,7,8,9],full:7,fulli:4,further:[2,3,9],g:[0,6,7,8],gamma:4,gdl:4,gener:[0,4,5,6,9],general_dic:9,generalis:4,generalizeddiceloss:4,get:9,get_dice_loss_modul:4,get_metric_nam:8,get_test_metr:9,get_train_metr:9,get_val_metr:9,github:[0,4,9],given:[3,4,6,8,9],glioblastoma:2,glioma:2,global:8,gpu:[0,7],grade:2,gradient:4,ground:[2,4],group:2,gz:3,h:4,ha:3,hausdorff95:[8,9],hausdorff:[4,8],hausdorff_dist:4,hausdorffdist:4,have:[2,4,8],healthi:2,heatmap:0,heatmaps_per_iter:0,height:4,held:2,helper:12,highli:4,holdout:2,hook:9,hot:4,how:[0,4],hp_optimis:7,html:[4,9],http:[0,4,9],hyperparamet:7,i:4,id:[3,8,11],id_to_class_nam:[3,8],ignor:4,ignore_index:4,ignore_nan_in_reduct:8,ignore_valu:4,imag:[3,4,6,8,9],image_id:8,implement:[0,4,9],improv:11,in_channel:9,includ:[2,4],include_background:4,include_background_in_reduced_metr:8,inclus:2,increas:0,index:[4,5,8,9],indic:[3,4,8],infer:6,inferenc:[1,10],inference_imag:6,inference_scan:6,inform:5,init_featur:9,initi:[0,3],initial_epoch:0,initial_training_set_s:3,initialis:7,input:[4,9,12],input_dimension:9,insert:4,institut:2,instruct:4,integ:8,intern:8,introduction_guid:9,io:[4,9],issu:[0,4],item:[0,3,11],items_to_label:[0,11],iter:[0,7,8,9],itself:0,jorg:4,keep:3,kei:[8,11],kera:3,kwarg:[0,3,9,11],l:4,label:[0,2,3,4,7,8,9,11],label_item:3,lambda:4,laplacian:4,latest:[4,9],layer:[4,8,9],lear:9,learn:[0,3,4,7,9],learning_r:[7,9],level:[7,8,9],li:4,lightn:[0,9],lightningdatamodul:3,lightningmodul:9,likelihood:4,list:[2,3,4,8,9,11,12],load:3,locat:4,log:[4,9],logger:0,logic:6,loop:9,loss:[0,1,7,9,10],loss_config:9,lower:2,lr_schedul:[0,7,9],m:4,mai:[8,12],main:[6,10],manag:3,map:[3,8],mask:[4,9],mask_filter_valu:3,mask_join_non_zero:3,master:9,match:9,mateuszbuda:9,math:4,max:4,maximum:4,mean:[4,8],mean_dice_score_0:7,medic:4,memori:3,merg:3,method:[4,8,9],metric:[1,8,9,10],metric_track:[1,10],metrics_to_aggreg:8,milletari:4,min:4,minimum:[4,7,9],modal:3,mode:[3,9],model:[0,1,3,4,6,7,8,10,11],model_config:7,model_selection_criterion:[0,7],modul:[0,3,4,5,6,8,9,10,11],monai:4,more:0,move:[2,3,7,9,12],mri:2,mrt:8,mulit:9,multi:[3,4,8,9],multi_label:[3,4,8,9],multilabel:[4,9],multimod:2,multimodalbraintumorsegment:2,multipl:[3,4,8,9,12],must:[4,8],mutual:2,n:[4,8],name:[2,3,6,7,8,9],nassir:4,navab:4,necessari:0,necrot:2,need:[4,8],neg:4,net:[4,7,9],network:4,neural:4,neuroradiologist:2,next:11,nii:3,nli:4,nllloss:4,nn:[4,9],non:[2,3],none:[0,3,4,6,7,8,9],normal:4,note:[4,8],np:11,num_class:[3,4],num_level:[7,9],num_work:[3,7],number:[0,3,4,7,8,9,11],object:[0,6,7,9],often:0,one:[3,4,7,8,9],onli:[0,7,8,9],oper:[0,7],optim:9,optin:9,option:[0,3,4,7,8,9,11,12],org:9,origin:[4,9],ot:3,other:[0,7],otherwis:[4,8],our:2,ourselin:4,out:2,out_channel:9,output:[4,8,9],output_dir:12,over:4,overlap:4,overwritten:3,p:4,p_:4,p_dropout:9,packag:4,page:[4,5],parallel:[0,7],param:[3,6,7,9,11],paramet:[0,3,4,7,8,9,11,12],pascalvocdatamodul:[1,10],pass:[3,4,7,9],path:[3,7,12],patholog:2,patient:2,pdf:9,per:[0,4,8],percentil:4,perform:4,pin_memori:3,pipelin:[0,7],pixel:4,point:[4,8],posit:4,pre:[2,4],precis:9,predicit:9,predict:[4,6,7,8,9],predict_step:9,prediction_count:[6,7],prediction_dir:[6,7],probabl:[4,9],process:2,produc:9,project:7,proper:[0,2],properti:9,protocol:2,provid:[0,2,4,7,8],pseudo:[3,11],pseudo_label:3,publish:2,py:9,pytorch:[0,3,4,9,11],pytorch_fcn_resnet50:9,pytorch_lightn:[0,3,9],pytorch_model:9,pytorch_u_net:9,pytorchfcnresnet50:[1,10],pytorchmodel:[0,1,10],pytorchunet:[1,10],queri:[0,7,11],query_strategi:[1,10],querystrategi:[0,1,10],r_:4,random:[3,7,12],random_sampl:3,random_st:[3,7,12],randomli:2,rate:[0,7,9],re:2,readi:8,readthedoc:[4,9],receiv:2,recurs:9,reduc:4,reducelronplateau:[0,7,9],reduct:[4,8],reduction_across_class:8,reduction_across_imag:8,region:2,regist:4,relat:8,remove_wandb_cach:0,repres:[2,3,4,8,11],reproduc:3,requir:9,reset:[0,4,8,9],reset_paramet:9,reset_weight:0,resnet50:9,resnet:9,resolut:2,run:[0,3,6,7,12],run_active_learning_pipelin:7,run_active_learning_pipeline_from_config:7,run_experi:10,run_script:12,s:[2,4,9],same:[4,8],sampl:[3,5,9],save:[0,7,12],save_model_every_epoch:[0,7],sbatch:12,sbatch_dir:12,sbatch_finished_dir:12,sbatch_run_dir:12,scalar:4,scan:[2,3,6,8],scanner:2,scatter:[4,8],schedul:9,scipi:4,score:[4,8],script:12,search:5,sebastien:4,second:8,see:[0,4,8,9],segment:[4,7,8,9],segmentationloss:4,segmentationmetr:4,sei:4,select:[0,2,3,11],select_items_to_label:11,sensit:[4,8,9],separ:[8,12],sequenc:3,set:[2,3,7,8,9],setup:[2,3,9],setup_train:0,sever:[2,8],sgd:9,shape:[4,8],share:2,sharp:8,should:[0,3,4,7,9,11,12],shuffl:[3,7],sigmoid:[4,8,9],similar:4,simpl:4,simul:[0,7],sinc:[0,2],singl:[3,4,8,9],single_class_hausdorff_dist:4,size:[0,3,4,5,7,8,9],skull:2,slice:[3,4,8],slice_id:[4,8],slices_per_imag:[4,8],smooth:4,so:2,softmax:[4,8,9],solid:2,some:[3,4],sourc:[0,3,4,6,7,8,9,11,12],space:4,specif:[0,3,4,7,8,9,11],specifi:[4,7,9,12],speed:[3,4],split:[3,5],spot:2,squar:4,stage:[3,8,9],start:[7,9,12],start_sbatch_run:12,starter:9,state:[3,4,8,12],step:[2,7,9],stop:[0,7],store:7,str:[0,3,4,7,8,9,11],strategi:[0,7,11,12],strategy_config:[7,12],string:[0,3,4,7,8,9,12],strip:2,structur:[2,3],subclass:3,subject:2,subpackag:5,subset:[4,8,11],sudr:4,sum:4,sum_:4,superclass:11,support:4,sweep:7,tag:7,take:[4,8,9,12],taken:[2,4,8,9],target:[4,8],task:[2,4,8,9],tcia:2,tensor:[4,8,9],term:4,test:[3,9],test_dataload:3,test_epoch_end:9,test_metr:9,test_metric_confidence_level:9,test_set_s:3,test_step:9,th:4,them:8,therefor:4,thi:[0,3,4,7,8,9,12],three:2,thu:[4,8],time:0,tn:4,tom:4,torch:[4,9],torchmetr:[4,8],torchvis:9,total:2,toward:[7,9],tp:4,track:8,train:[0,2,3,4,7,9],train_dataload:3,train_metr:9,train_metric_confidence_level:9,trainer:[0,9],training_epoch_end:9,training_set_s:3,training_step:9,transform:4,truth:[2,4],tumor:2,tupl:[3,11],turn:4,type:[0,3,4,7,8,9,11],typic:4,u:[2,7,9],u_net:[7,9],unbalanc:4,unet:[1,10],uniform:4,union:[4,8],unlabel:[2,3,11],unlabeled_dataload:3,unlabeled_set_s:3,until:0,up:[3,4],updat:[0,4,7,8,9],us:[0,3,4,6,7,8,9],use_amp:9,usual:9,v:4,val:9,val_dataload:3,valid:[2,3,9],validation_epoch_end:9,validation_set_s:3,validation_step:9,valu:[3,4,8,9,12],vercauteren:4,version:4,vision:9,volum:4,volumetr:4,voxel:4,w:7,w_l:4,wa:[2,4,9],wandb:0,wandb_project_nam:7,we:2,weigh:4,weight:[0,4,7,9],weight_typ:4,well:[2,4],wenqi:4,were:[2,4],when:[0,7,9,12],where:[0,4,7,8],whether:[0,3,4,7,8,9,12],which:[3,4,7,8,9,12],who:2,whole:[0,2,3,7,8],whose:[4,8],width:4,worker:[3,7],wrap:[6,9],wrapper:4,x:[4,8,9],y:[4,8],your:9,z:8,zero:[3,4]},titles:["active_learning","active_learning_framework","The BraTS Dataset","datasets","functional","Welcome to Active Segmentation\u2019s documentation!","inferencing","main module","metric_tracking","models","src","query_strategies","run_experiments module"],titleterms:{"function":4,The:2,activ:5,active_learn:0,active_learning_framework:1,brat:2,bratsdatamodul:3,bratsdataset:3,combined_per_epoch_metr:8,combined_per_image_metr:8,content:5,data:2,datamodul:3,dataset:[2,3],detail:2,document:5,gener:2,indic:5,inferenc:6,inform:2,loss:4,main:7,metric:4,metric_track:8,model:9,modul:[7,12],pascalvocdatamodul:3,pytorchfcnresnet50:9,pytorchmodel:9,pytorchunet:9,query_strategi:11,querystrategi:11,run_experi:12,s:5,sampl:2,segment:5,size:2,split:2,src:10,subpackag:1,tabl:5,unet:9,welcom:5}})