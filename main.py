# import pre_processing_grahams
# import pre_processing_CLAHE
import  Adaptive_Median_Filter
import pre_processing_kaggle
if __name__ == '__main__':

    pre_processing_kaggle.preprocessing('pre_processing_4/messidor-2/500/', 'pre_processing_4_to_0/messidor_2/')
    pre_processing_kaggle.preprocessing('pre_processing_4/idrid_datast/500/test/', 'pre_processing_4_to_0/IDRID/test/')
    pre_processing_kaggle.preprocessing('pre_processing_4/idrid_datast/500/train/', 'pre_processing_4_to_0/IDRID/train/')
    pre_processing_kaggle.preprocessing('pre_processing_4/APTOS/500/', 'pre_processing_4_to_0/APTOS/')

    pre_processing_kaggle.preprocessing('pre_processing_4/exp_2_b/500/', 'pre_processing_4_to_0/exp_2_b/')
    pre_processing_kaggle.preprocessing('pre_processing_4/exp_2_c/500/', 'pre_processing_4_to_0/exp_2_c/')
    pre_processing_kaggle.preprocessing('pre_processing_4/exp_2_d/500/', 'pre_processing_4_to_0/exp_2_d/')
