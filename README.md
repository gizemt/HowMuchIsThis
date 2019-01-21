# Product Price Estimation from Images

This project (and the materials) is a result of a joint work with Subhro Roy.

There might be some times, for example, you might be willing to buy an item at flea market and want to have an idea about the actual worth of the item to make sure that you are not overpaying; you want to set a reasonable price for an item you would want to sell; you could see a product, a bag, a coat, a hat, on someone but cannot ask for its price. Even in Pinterest, you might like an item but cannot see the original or approximate price of that item. Inspiring from these problems, we wanted to develop a system that solely takes an image as the input, and outputs the estimated price of that item. 

The concept of estimating the price of an item whose worth is otherwise unknown using different attributes is a well-established research subject in many areas including economics, statistics and machine learning applications. In economics, it is usually done deterministically by thoroughly analyzing various aspects of the product such as its cost, market price and/or demand [3]. In statistics and machine learning, regression analysis is a commonly used tool given some quantitative and qualitative attributes of the products [1]. However, there is no established research or application in the literature that benefits from images as attributes or that tries to estimate the price of an item using its image.

We used a subportion of Amazon product dataset obtained from Asst. Prof. Julian McAuley, UCSD [5]. Our implementations through the project can be combined under two approaches: One is trying to estimate the price directly from Image features without using any other extra information. Second is trying to benefit from the categories of the products and trying to estimate the category first, and then the price.

Without using any other external information such as the brand or quality of a product, estimating the exact price by just using the image is a difficult goal to accomplish with high accuracy. In order to both scale down the problem to the level of our available computational power and hopefully ease the problem a bit, we focused only on Home&Kitchen subcategory. The reason we chose this specific subcategory is we intuitively thought that it would have less products that has very similar appearance but very different internal specifications that affects the price, such as two laptops having different RAMs, CPUs or HDD/SSD drives as in Electronics category. It is obvious that we cannot eliminate this problem entirely by just choosing an appropriate subcategory; but at least we tried to choose the one that might be affected less by this problem.

In order to tackle the problem, we used features extracted by a BVLC CaffeNet with 5-layer CNN pre-trained on ImageNet [5]. On these features, we applied different regression methods (kNN, neural nets and decision trees) to estimate (first categories, and then) prices from images. We achieved ~47% accuracy whereas a naive guess returned ~2%, accuracy being within $10 bracket of the actual price.

<p align="center">
  <img src="/Figures/histpred.pdf" width="350" title="Histogram of real and predicted prices">
  <img src="/Figures/histpred_100.pdf" width="350" title="Histogram of real and predicted prices">
</p>

There were several reasons leading to low accuracy:

1. Amazon product dataset is a huge dataset. It has various types of noise and categories are not accurately defined in the dataset. We expect the results to improve by using a pre-trained object recognition network first, classifying the objects more accurately, and then estimating the prices with the help of that information.

2. Second challenge is the nature of the problem itself. Same type of products might have very different prices based on their quality, brand or designer. By just looking at the images, there is no chance to identify and eliminate these factors, assuming we are not implementing text recognition on the image in order to extract the brand. Using text recognition on the image might work to some extent if the objects have visible and recognizable brand names on them. Another solution is to use product descriptions available in the metadata.

3. The product price distribution was highly biased. There were many products in $0-$50 range, and very few outside of that range. This bias is reflected to the results, and balancing the dataset might improve the resulting accuracy.

