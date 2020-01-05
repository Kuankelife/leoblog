---
layout: post
title:  "Support Vector Machine Methods on Classification"
subtitle: ''
date:   2020-01-01
author: "YU"
header-style: text
<!-- header-img: "img/yu-img/post-img/post_head/keyibukeyi1.jpg"
header-mask: 0.4 -->
tags:
  - 机器学习
  - 支持向量机
mathjax: True
---
<iframe frameborder="no" border="0" marginwidth="0" marginheight="0" width=330 height=86 src="//music.163.com/outchain/player?type=2&id=1332566421&auto=1&height=66"></iframe>

<p>"//music.163.com/outchain/player?type=2&id=1332566421&auto=1&height=66"</p>

![2401](http://up.desktx.net/pic/c9/3b/ef/c93bef6133d4348f9d9f1d9316865835.jpg)


# Abstract

Support vector machine (SVM) is a machine learning method based on the VC dimension theory and the principle of structural risk minimization, which aims to solve quadratic problems. It has become the research field of machine learning because of its excellent performance  on classification. This review introduces the theoretical basis of support vector machine and its algorithms like chunking algorithm, decomposition algorithm, sequential minimal optimization， incremental algorithm, least squares support vector machine, fuzzy support vector machine, and granular support vector machine, twin support vector machines, ranking support vector machines. We summarize the difference and discuss the advantages and disadvantages of these SVM algorithms. Finally, we analyze current  development status as well as point out its future advanced research direction. 


Keywords: Support vector machines, Statistical learning theory, Classification


# Introduction
Support vector machines was proposed by (Cortes and Vapnik, 1995). It has unique advantages in solving small sample classiﬁcation, nonlinear and high-dimensional pattern recognition as well as other machine learning problems such as function ﬁtting. The support vector machine is based on the VC dimension theory of statistical learning theory and the principle of structural risk minimization, and seeks the best between the model complexity and the learning ability according to limited sample data information. in order to obtain the best generalization ability of model. In addition, it has a solid mathematical theoretical foundation and clear model. It can ensure that the extreme value solution found is a global optimal solution instead of a local minimum solution, which can be overcome the common problems in machine learning tasks like dimensional disaster and over-ﬁtting problems. These mathematical features also determine that the SVM method has a better generalization ability for unknown sample data sets. Because of these advantages, SVM can be well applied to the ﬁelds of pattern recognition, probability density function estimation, time series prediction, regression estimation and other ﬁelds. It has also been widely used in pattern recognition for handwritten digit recognition (Mustafa andDoroslovacki,2004),textclassiﬁcation(TongandKoller,2001),imageclassiﬁcationandrecognition(Chapelleetal., 1999), gene classiﬁcation (Guyon et al., 2002) and time series prediction (Müller et al., 1997). This review will ﬁrst introduce the theory of support vector machines systematically, and then review the current SVM training algorithms. Based on this, we will analyze the shortcomings of SVM methods and looks forward to future research directions.


# Support vector machine theory

## VCdimensiontheory


Machine learning obtain new knowledge and skills by fitting data into its model, using training algorithm of model to simulate or realize human learning capabilities. There are three main means to implement machine learning: statistical prediction methods, empirical nonlinear methods, and statistical learning theory. Support vector machine is a machine learning method of statistical learning theory (SLT) and it studies the rules of small samples (Ying xin,Xiao gang,2005). This SLT theory establishes a new theoretical system for small sample statistics. And SVM is based on the VC dimension theory (Kearnsetal.,1994) of SLT and the principle of structural risk minimization. For an indicator function set, if there are $h$ samples that can be separated by the function in the function set according to all possible ${2}^{h}$ forms, then this function set can be broken up using this h samples, and the VC dimension of the function set is the largest number of samples $h$. The VC dimension reflects the learning ability of the function set. The larger the VC dimension, the more complicated the learning machine, which means the stronger the learning ability.

## Principleofstructuralriskminimization
For principle of structural risk minimization, SLT describes the concept of generalized error bound, which states that the actual error of machine learning is consist of empirical risk and confidence risk. For all functions in the indicator function set, the empirical risk and the actual risk satisfiy the following relation with a probability of at least $1-\eta$:
\begin{equation}
R(\boldsymbol{w}) \leqslant R_{\mathrm{Emp}}(\boldsymbol{w})+\sqrt{\frac{h\left(\ln \frac{2 n}{h}+1\right)-\ln \frac{\eta}{4}}{n}}
\end{equation}
Where h is the VC dimension of the function set; n is the number of samples. So the generalized error bound is


\begin{equation}
R(\boldsymbol{w}) \leqslant \operatorname R_{Emp}(\boldsymbol{w})+\varphi(n / h)
\end{equation}
Where $ R (w) $ is the actual risk, $ remp (w) $ is the empirical risk, and $ φ (\ fac {n} {h}) $ is the confidence risk. Confidence risk is related to two variables, one is sample size, the larger the sample size, the more likely the machine learning results are correct, and the smaller the confidence risk. The other one is the VC dimension of the classification function, the worse the ability to change, the greater the confidence risk. The goal of SVM is try to Find the minimum of the sum of empirical risk and confidence risk, that is called structural risk minimization (SRM).

## Theory foundation
SVM try to seek a way to deal with two types of data classification problems. Its goal is to find a hyperplane, so that the points of different classes in the training sample dataset fall on both sides of the hyperplane, and at the same time, the marginal space on both sides of the hyperplane is required to be maximized. For two-dimensional and two types of linearly separable data, the support vector machine can theoretically achieve optimal classification, and can be extended to high-dimensional space. The optimal classification place is called the optimal hyperplane. For two-dimensional two-class data classification, given training samples
$D_{i} =\left(x_{i}, y_{i}\right), i=1, \cdots, l, y_{i} \in\{+1,-1\}$, where $x_{i}$ is training sample， $y_{i}$ is label set of two classes 0 and 1, the hyperplane $w x+b=0$，the interval from the points to the hyperplane is
$\delta_{i}=\frac{1}{\|w\|}\left|g\left(x_{i}\right)\right|$, 
We expect that the training samples can be separated correctly and the interval should be maximized. This two-class classification problem is transformed into a constrained minimum problem:

$$
\min \frac{1}{2}|w|^{2}
$$



$$
\text {subject to} y_{i}\left(w x_{i}+b\right)-1 \geqslant 0 \quad i=1, \cdots, l
$$

When data is linearly inseparable, we add a variable $\xi_{i} \geqslant 0$ and a penalty factor $C$ to condition (4).
\begin{equation}
\min \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{l} \xi_{i}
\end{equation}

$$subject\ to\ y_{i}(w x_{i}+b) \geqslant 1-\xi_{i}$$
\begin{equation}
i=1, \cdots, l,\xi_{i} \geqslant 0
\end{equation}
$C$ is a constant. Now using the Lagrange multiplier method to solve the equations:
\begin{equation}
L=\frac{1}{2}\|w\|^{2}-\sum_{i=1}^{l} \alpha_{i} y_{i}\left(w \cdot x_{i}+b\right)+\sum_{i=1}^{l} \alpha_{i}
\end{equation}
$$\alpha_{i}\geqslant 0$$
The optimal classification function obtained after solving the above problem is

$$
f(x)=\operatorname{sgn}\left\{\left[\sum_{j=1}^{l} \alpha_{j}^{*} y_{j}\left(x_{j} \cdot x_{i}\right)\right]+b^{*}\right\}
$$

<div align = 'center'> <img src = 'https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1575726033&di=a6755d93c25e47e870d3d5d35ba1dae0&imgtype=jpg&er=1&src=http%3A%2F%2Fblog.nsfocus.net%2Fwp-content%2Fuploads%2F2016%2F12%2FSVM%25E5%258E%259F%25E7%2590%2586.jpg'></div>

# Support vector machine algorithm
## Chunking algorithm
The Chunking algorithm (Boser et al., 1992) is to iteratively delete the rows and columns with zero Lagrange multipliers in the matrix through the KKT condition, and retain the non-zero support vector part. From Figure 1, we can know that for a given sample data, the support vector is small, so that by continuously iterating a large quadratic programming problemintosmall-scalequadraticprogrammingproblems,reducingthesamplespacetosupportvectorspace,thenumber of samples is reduced, thereby reducing the storage capacity requirements of the computer during the training process. Speeded up the training speed, which is ultimately affected by the number of support vectors. 


Chunking algorithm reduces the size of the matrix from the square of the number of training samples to the square of the number of samples with non-zero Lagrange multipliers, which greatly reduces the storage capacity requirements of the training process. Chunking algorithm can greatly improve the training speed, especially when the number of support vectors is much smaller than the number of training samples. However, if the number of support vectors is large, as the number of iterations of the algorithm increases, the selected block will become larger and larger, and the training speed of the algorithm will still become very slow.


## Incremental algorithm
Incrementallearning(Syedetal.,1999)isthatwhenamachinelearningsystemprocessesnewsamples,itcanadd,modify, or delete only those parts of the original learning results that are related to the new samples, and parts that are not related are not touched. An outstanding feature of the incremental training algorithm is that the learning of the support vector machine is not performed ofﬂine at one time, but a process of adding data one by one to iterative optimization one by one. Only a small batch of data that can be processed by the conventional secondary algorithm at a time is selected as the increment, and the support vector in the original sample and the newly added sample are mixed for training until the training samples are used up. (Cauwenberghs and Poggio, 2001) proposed the exact solution of incremental training, which is the effect of adding a training sample or reducing a sample on the Lagrange coefﬁcient and support vector. (Kong et al., 2006) proposed an incremental motion vector machine based on the center distance ratio, which uses the center distance ratio to improve the convergence speed without ensuring that the accuracy of training and testing has not changed.



## Decomposition algorithm
Decompositionalgorithm(Osunaetal.,1997)iscurrentlythemainmethodtoeffectivelysolvelarge-scaleproblems. The decomposition algorithm decomposes the secondary programming problem into a series of smaller secondary programming sub-problems, and solves iteratively. In each iteration, a subset of the Lagrangian multiplier component is selected as the working set, and a traditional optimization algorithm is used to solve a subproblem of the quadratic programming. Taking classiﬁcation SVM as an example, the main idea of the decomposition algorithm is to divide the training samples into working set B and non-working set N. The number of samples in working set B is q, and q is far less than the total number of training samples. Training is performed on the samples in working set B at a time, and the training samples in N are ﬁxed. The key of this algorithm is to choose an optimal working set selection algorithm, and a random method is used in the selection of the working set, so the convergence speed of the algorithm is limited.


## Least Squares support vector machine
When solving large-scale QP problems, the Chunking algorithm and decomposition algorithm in the basic support vector machine algorithm will have problems such as dimensional disaster and slow solution speed. In the process of solving the support vector machine, the constraints are inequality constraints. In order to simplify optimization The process and ensure a certain learning speed, use the equality constraint to replace the inequality constraint in equation (5), and use the leastsquareslossfunctioninsteadoftheinsensitivelossfunctiontosimplifythesolutionprocess,soastoobtaintheleast squares support vector machine algorithm (Suykens and Vandewalle, 1999). 


\begin{array}{l}{\min L(\omega)=\frac{1}{2}\|\omega\|^{2}+\frac{c}{2} \sum_{i=1}^{n} \xi^{2}} \\ {\text { s.t. } y_{i}\left(\omega^{T} \Phi\left(x_{i}\right)+b\right)=1-\xi, \xi>0}\end{array}


## Granular support vector machines
Granularsupportvectormachine(Tangetal.,2004)addsgranularcomputingintothelearningalgorithmofsupportvector machine, divides granularity through association rules and clustering, etc., constructs information granules in granular space, and then uses the information on the information granules to obtain the SVM objective function. At present, the main research on granular partitioning is: granular support vector machines based on association rules(Tang et al., 2005), mining frequent patterns of sample data sets using association rules, segmenting the sample feature space to establish a granularspace,andﬁnallytrainingonthegranularfeaturespaceLearn. Clustering-basedgranularsupportvectormachine uses clustering method to partition the data in the sample space, and then selects the particles with more information to learn the sample information, thereby achieving classiﬁcation or regression problems.


## Fuzzy support vector machines
In order to overcome the inﬂuence of noise and outlier points on support vector machines, (Li and Shu, 2008) combined fuzzy mathematics and support vector machines to propose fuzzy support vector machines, which are mainly used to process noise data in training samples. . The main idea is to aim at the sensitivity of the support vector machine to the noise and outliers in the training sample, add a degree of membership to the training sample set, and give the support vectorahigherdegreeofmembershipinsteadofthesupportvectorandnoiseoutliers. Givesmallermembershiptoreduce the inﬂuence of non-support vectors, noise and outliers on the superior hyperplane. The problem in FSVM is how to determine the membership value, that is, how to determine the weight of each sample.(Li and Shu, 2008) proposed a method of determining membership based on class center, using a linear function of the distance from the sample point toitsclasscenterasthe membershipfunction, butthemembershipfunctionwoulddependheavilyonthegeometry ofthe training sample set, reducing the membership of the support vector degree.



## Twin support vector machines
(Khemchandani et al., 2007) proposed a classiﬁer of binary data-twin support vector machines . TWSVMs are similar in form to traditional support vector machines (Kumar and Gopal, 2009). They not only have the advantages of traditional support vector machines, but also have better processing capabilities for large-scale data. TWSVMs get a classiﬁcation plane for each of the two classes(Peng, 2010), and the data belonging to each class is surrounded as much as possible around the corresponding classiﬁcation plane. Then TWSVMs construct a classiﬁcation hyperplane by optimizing a pair of classiﬁcation planes. That is to say, TWSVMs need to solve a pair of QP problems, while SVM is to solve a QP problem, but in TWSVMs, the data of one class must be used as a constraint for another QP problem, and vice versa. The existing twin model does not have a characteristic similar to the traditional support vector machine, namely the interval. Therefore, if the advantages of the twin model and the traditional support vector can be successfully combined, a twin support vector machine model with both a faster training speed and a better theoretical basis can be obtained.


The existing twin model does not have a characteristic similar to the traditional support vector machine, namely the interval. Therefore, if the advantages of the twin model and the traditional support vector can be successfully combined, a twin support vector machine model with both a faster training speed and a better theoretical basis can be obtained.

# Conclusion
In the case of limited samples, SVM is a general effective method for machine learning, which shows superior performances in theory and practical applications. The superiority of SVM has made it a great development in the ﬁelds of pattern recognition, regression analysis, function estimation, time series prediction and so on. Both the theory and algorithmresearchofSVMhavemadegreatprogress,however,therearealsomanyshortcomings. Inthepracticalapplication oftrainingdata,therearestillproblemssuchascalculationspeedandstoragecapacity,whichacquirefurtherdevelopment and improvement. The possible research directions includes: further improving the algorithm of SVM. The core of the support vector machine algorithm is the kernel function and its parameters. Their correct selection has a great impact on the prediction and generalization performance of the SVM. For a speciﬁc research problem, which kernel function to choose and ﬁnd the optimal parameters is very important to solve the problem. Therefore, how to quickly and accurately select the kernel function and the corresponding parameters to meet the requirements of fastness and accuracy is the problem to be solved; in addition, the learning efﬁciency of SVM depends on the size of the sample dataset, the training efﬁciency of a large-scale sample data set cannot reach the ideal training efﬁciency. Therefore, the further improvement of the training efﬁciency and generalization performance of the SVM algorithm is inevitable for improvement of the algorithm itself. The second direction exploring the integration of SVM with ﬁelds. In recent years, new support vector machines, such as FSVM, GSVM, and TWSVM, constructed by the integration of support vector machines and other ﬁelds, have improved in training efﬁciency and generalization performance, but the existing models have also differed. Finally, how to effectively extend the two-class classiﬁer to multi-class problems is still a future research of SVM.

# References

Bernhard E Boser, Isabelle M Guyon, and Vladimir N Vapnik. A training algorithm for optimal margin classiﬁers. In Proceedings of the ﬁfth annual workshop on Computational learning theory, pages 144–152. ACM, 1992.


Gert Cauwenberghs and Tomaso Poggio. Incremental and decremental support vector machine learning. In Advances in neural information processing systems, pages 409–415, 2001.


Olivier Chapelle, Patrick Haffner, and Vladimir N Vapnik. Support vector machines for histogram-based image classiﬁcation. IEEE transactions on Neural Networks, 10(5):1055–1064, 1999.


Corinna Cortes and Vladimir Vapnik. Support-vector networks. Machine learning, 20(3):273–297, 1995.

Isabelle Guyon, Jason Weston, Stephen Barnhill, and Vladimir Vapnik. Gene selection for cancer classiﬁcation using support vector machines. Machine learning, 46(1-3):389–422, 2002.


Michael J Kearns, Umesh Virkumar Vazirani, and Umesh Vazirani. An introduction to computational learning theory. MIT press, 1994.


R Khemchandani, Suresh Chandra, et al. Twin support vector machines for pattern classiﬁcation. IEEE Transactions on pattern analysis and machine intelligence, 29(5):905–910, 2007.


Bo Kong, Xiao-mao Liu, and Jun Zhang. Incremental support vector machine based on center distance ratio. Journal of Computer Applications, 26(6):1434–1436, 2006.


M Arun Kumar and Madan Gopal. Least squares twin support vector machines for pattern classiﬁcation. Expert Systems with Applications, 36(4):7535–7543, 2009.


Xuehua Li and Lan Shu. Fuzzy theory based support vector machine classiﬁer. In 2008 Fifth International Conference on Fuzzy Systems and Knowledge Discovery, volume 1, pages 600–604. IEEE, 2008.


K-RMüller,AlexanderJSmola,GunnarRätsch,BernhardSchölkopf,JensKohlmorgen,andVladimirVapnik. Predicting timeserieswithsupportvectormachines. In International Conference on Artiﬁcial Neural Networks, pages999–1004. Springer, 1997.


Hussam Mustafa and Milos Doroslovacki. Digital modulation recognition using support vector machine classiﬁer. In Conference Record of the Thirty-Eighth Asilomar Conference on Signals, Systems and Computers, 2004., volume 2, pages 2238–2242. IEEE, 2004.


Edgar Osuna, Robert Freund, and Federico Girosi. An improved training algorithm for support vector machines. In Neural networks for signal processing VII. Proceedings of the 1997 IEEE signal processing society workshop, pages 276–285. IEEE, 1997.


Xinjun Peng. A ν-twin support vector machine (ν-tsvm) classiﬁer and its geometric algorithms. Information Sciences, 180(20):3863–3875, 2010.


Johan AK Suykens and Joos Vandewalle. Least squares support vector machine classiﬁers. Neural processing letters, 9 (3):293–300, 1999.


Nadeem Ahmed Syed, Syed Huan, Liu Kah, and Kay Sung. Incremental learning with support vector machines. 1999.


Yuchun Tang, Bo Jin, Yi Sun, and Yan-Qing Zhang. Granular support vector machines for medical binary classiﬁcation problems. In 2004 Symposium on Computational Intelligence in Bioinformatics and Computational Biology, pages 73–78. IEEE, 2004.


Yuchun Tang, Bo Jin, and Yan-Qing Zhang. Granular support vector machines with association rules mining for protein homology prediction. Artiﬁcial Intelligence in Medicine, 35(1-2):121–134, 2005.


Simon Tong and Daphne Koller. Support vector machine active learning with applications to text classiﬁcation. Journal of machine learning research, 2(Nov):45–66, 2001.


Li Yingxin and Ruan Xiaogang. Feature selection for cancer classiﬁcation based on support vector machine. Journal of Computer Research and Development, 42(10):1796–1801, 2005.


