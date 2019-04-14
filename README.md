[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **Opinion Networks in Social Media** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

Author: Yegor Klochkov

## Description:

An integration of social media characteristics into an econometric framework requires modeling a high dimensional dynamic network with dimensions of parameter $\Theta$  typically much larger than the number of observations. 
To cope with this problem we impose two structural assumptions onto the singular value decomposition of $\Theta = U D V^{\top}$. Firstly, the matrix with probabilities of connections between the nodes of a network has a rank much lower than the number of nodes. Therefore, there is limited amount of non-zero elements on the diagonal of $D$ and the whole operator admits a lower dimensional factorisation. Secondly, in observed social networks only a small portion of users are highly-affecting, leading to a sparsity regularization imposed on singular vectors $V.$

Using a novel dataset of 1069K messages from 30K users posted on the microblogging platform StockTwits during a 4-year period (01.2014-12.2018) and quantifying their opinions via natural language processing, we model their dynamic opinions network and further separate the network into communities. With a sparsity regularization, we are able to identify important nodes in the network.

## Acknoledgements:

Financial support from the German Research Foundation (DFG) via International Research Training Group 1792 ”High Dimensional Non Stationary Time Series”, Humboldt-Universität zu Berlin, is gratefully acknowledged.
