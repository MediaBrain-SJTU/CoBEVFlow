# Notion for subfolder name

- npj: not project point cloud to ego coordinate first, that is, 1-round communication.

- pj: project point cloud to ego coordinate first, that is, 2-round communication. 

 The original paper considers a 2-round communication setting where each agent shares pose first and encodes feature map in receiver's coordinate by transforming its point cloud for to mitigating the discretization issue. It leads to higher performance but is less practical as each agent's computational cost is expensive when the number of collaborators goes up. I think it's more reasonable for each agent transmiting the same feature map to all collaborators in 1-round communication.