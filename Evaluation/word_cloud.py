from matplotlib import pyplot as plt
from wordcloud import WordCloud

textdictdis0={'previous':91,'rose':83,'good book':60,'show off':46,'spicy':43,'science fiction':23,'want to read':20,'coat':20,'soak':13,'cash':12}
textdictdis1={'teacher':48,'stadium':32,'watch the game':32,'classroom':11,'temporarily':16,'tender':16,'soak':16,'Crush':16,'highlight':16,'this season':16}
textdictdis2={'terror':45,'subtle':44,'red':29,'plastic surgery':24,'bathing':24,'close':22,'needle':17,'love':14,'price':14,'detoxification':14}
textdictdis3={'golden globe award':27,'judge':25,'chinese':17,'leader':16,'the film':14,'radio and television':13,'present':11,'chinese cabbage':9,'first place':15,'praise':6}

textdictemb0={'selection':23,'previous':23,'show off':21,'coma':21,'good book':21,'drinking tea':20,'spicy':20,'read book':19,'an article':18,'science fiction':17}
textdictemb1={'student':23,'book':21,'judge':20,'learned':21,'advertisement':20,'observed':19,'left foot':19,'academician':20,'draft fee':19,'advocate':17}
textdictemb2={'less or less':25,'fluent':24,'collagen':20,'apply face':20,'purchase price':19,'hair mask':19,'brighten':19,'smallpox':18,'skill':17,'close':14}
textdictemb3={'judge':20,'awards ceremony':17,'secretary general':16,'keep warm':15,'second':12,'million':9,'crown king':9,'explain':8,'rule':9,'fool':5}

fig, ax = plt.subplots()
plt.subplot(421)
wc = WordCloud(
			   background_color = 'white',
			   width = 1000,
			   height = 800,
			   ).generate_from_frequencies(textdictdis0)
wc.to_file('dis0.png')  # 保存图片
plt.imshow(wc)  # 用plt显示图片
plt.axis('off')  # 不显示坐标轴

plt.subplot(422)
wc = WordCloud(
			   background_color = 'white',
			   width = 1000,
			   height = 800,
			   ).generate_from_frequencies(textdictemb0)
wc.to_file('emb0.png')  # 保存图片
plt.imshow(wc)  # 用plt显示图片
plt.axis('off')  # 不显示坐标轴

plt.subplot(423)
wc = WordCloud(
			   background_color = 'white',
			   width = 1000,
			   height = 800,
			   ).generate_from_frequencies(textdictdis1)
wc.to_file('dis1.png')  # 保存图片
plt.imshow(wc)  # 用plt显示图片
plt.axis('off')  # 不显示坐标轴

plt.subplot(424)
wc = WordCloud(
			   background_color = 'white',
			   width = 1000,
			   height = 800,
			   ).generate_from_frequencies(textdictemb1)
wc.to_file('emb1.png')  # 保存图片
plt.imshow(wc)  # 用plt显示图片
plt.axis('off')  # 不显示坐标轴

plt.subplot(425)
wc = WordCloud(
			   background_color = 'white',
			   width = 1000,
			   height = 800,
			   ).generate_from_frequencies(textdictdis2)
wc.to_file('dis2.png')  # 保存图片
plt.imshow(wc)  # 用plt显示图片
plt.axis('off')  # 不显示坐标轴

plt.subplot(426)
wc = WordCloud(
			   background_color = 'white',
			   width = 1000,
			   height = 800,
			   ).generate_from_frequencies(textdictemb2)
wc.to_file('emb2.png')  # 保存图片
plt.imshow(wc)  # 用plt显示图片
plt.axis('off')  # 不显示坐标轴

plt.subplot(427)
wc = WordCloud(
			   background_color = 'white',
			   width = 1000,
			   height = 800,
			   ).generate_from_frequencies(textdictdis3)
wc.to_file('dis3.png')  # 保存图片
plt.imshow(wc)  # 用plt显示图片
plt.axis('off')  # 不显示坐标轴

plt.subplot(428)
wc = WordCloud(
			   background_color = 'white',
			   width = 1000,
			   height = 800,
			   ).generate_from_frequencies(textdictemb3)
wc.to_file('emb3.png')  # 保存图片
plt.imshow(wc)  # 用plt显示图片
plt.axis('off')  # 不显示坐标轴

plt.show()  # 显示图片