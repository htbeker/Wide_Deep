推荐系统与深度学习的结合。在推荐系统中，记忆体现的准确性，而泛化体现的是新颖性，wide_deep能将两者相结合。

1、 Memorization 和 Generalization
    这个是从人类的认知学习过程中演化来的。人类的大脑很复杂，它可以记忆(memorize)下每天发生的事情（麻雀可以飞，鸽子可以飞）然后泛化(generalize)这些知识到之前没有看到过的东西（有翅膀的动物都能飞）。 
但是泛化的规则有时候不是特别的准，有时候会出错（有翅膀的动物都能飞吗）。那怎么办那，没关系，记忆(memorization)可以修正泛化的规则(generalized rules)，叫做特例（企鹅有翅膀，但是不能飞）。

    这就是Memorization和Generalization的来由或者说含义。
2、 Wide & Deep模型
    实际上，Wide模型就是一个广义线性模型，Deep就是指Deep Neural Network。Wide Linear Model用于memorization；Deep Neural Network用于generalization。
    同时训练Wide模型和Deep模型，并将两个模型的结果的加权和作为最终的预测结果。
    Wide模型：FTRL
    Deep模型：AdaGrad
