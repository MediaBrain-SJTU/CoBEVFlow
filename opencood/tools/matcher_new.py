class TrajectoryPrediction():
    def __init__(self):
        super().__init__()
    
    def forward(self, input_dict):
        """
            input_dict: 
            {
                [batch_id] : {
                    'ego' : {
                        'past_k_time_diff' : len=k, float
                        [0] {
                            pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                            pred_box_center_tensor : n, 7
                            scores: (n, )
                        }
                        ... 
                        [k-1]
                    }
                    'cav_id' : { ... }
                }
            }  
            output_dict:
            {
                [batch_id] : {
                    'ego' : {
                        'past_k_time_diff' : 
                        [0] {
                            pred_box_3dcorner_tensor: The prediction bounding box tensor after NMS. n, 8, 3
                            pred_box_center_tensor : n, 7
                            scores: (n, )

                        }
                        ... 
                        [k-1]
                    }
                    [cav_id] : { ... }
                }
            }
        """
        return 0