---
tags: tensorflow, keras
---
[TOC]

# Keras 大神歸位 - 深度學習

## 定義
++**普適化 (Generalization):**++
>Generalization refers to your model's ability to adapt properly to new, previously unseen data, drawn from the same distribution as the one used to create the model.

++**密集抽樣 (dense sampling):**++
>指訓練資料應該密集地涵蓋整個輸入空間的流形，尤其是在決策邊界(decision boundary)附近。

++**資訊洩漏 (information leak):**++
> 每次根據模型在驗證集上的表現，進而調整模型的超參數時，和驗證資料相關的一些資訊就會洩漏到模型中。如果僅調整一次，則只會洩漏極少量的資訊，此時驗證集尚可保存評估模型的可靠性。但是如果重複進行多次調整，那麼模型也可能對驗證資料過度配適 (overfitting to the validation set)。

++**抽樣偏差 (sampling bias):**++
> 當資料欠缺代表性時會發生抽樣偏差，其根源在於資料收集的方式與某些要預測的事物產生關連，進而導致資料內容有所偏差。

++**目標值洩漏(target leaking):**++
> 即訓練資料中的特徵提供了目標值的資訊，但這些資訊在實際應用場景中卻無法取得。

## 流形假說 manifold hypothesis
++**甚麼是流形?**++
所謂的流形(manifold)就是由原始空間中某些"線性子空間(subspace)"所形成的低維度子空間。

++**流形假說表示:**++
1. 機器學習模型只需擬合(學習)其輸入空間中的潛在流形(latent manifold)即可，在這些潛在子空間中的資料相對比較簡單、低維度、且具有高結構性。
2. 在這些流形中，兩個樣本之間必然可以進行內插(interpolate)，也就是說，可以沿著一條連續的路徑將一個樣本漸變成另一個樣本，而該路徑上所有的樣本都位在流形內。

:::info
樣本之間可以內插的特性，就是理解深度學習中普適化(generalization)能力的關鍵。
:::

---
++**關於內插法:**++
- 如果處理的資料可進行內插，那麼就可以藉由將新資料點連結到流形上相近的其他點，進而理解這些從未見過的資料點。也就是說可以靠空間的有限樣本來理解空間的整體性(totality)，作法就是利用內插法來填滿其中的空白。
- 內插法只能幫助我們理解那些與"已見過的事物**非常相近**"的對象，這個性質讓我們可以實現局部普適化(local generalization)。
- 對於那些與"已經見過的事物**不太相近**"的對象，其實也有可能普適化。
- 沿著模型學習到的曲線上移動，就會和在真實資料流形上移動的狀況很接近。如此一來，模型就可以透過在既有的訓練資料點之間進行內插，處理未曾見過的輸入。


## 深度學習
++**深度學習特性:**++
- 模型會在輸入與輸出之間建立一個連續且平滑的映射關係。這是因為模型必需是可微分，否則無法進行梯度下降。這個平滑的性質有助於逼近潛在流形
- 設計良好的深度學習模型可以結構化地映射訓練資料的資料形狀。
- 普適化能力主要取決於資料的自然結構，而非模型的特性。
- 只有當資料流形中的點可以內插化，才有可能進行普適化。

:::warning
資料中的特徵越清晰、越沒有雜訊干擾，普適化的能力越好，這是因為輸入空間變得更簡單且有著較佳的結構。
:::
- 若要模型表現良好，最好可以在輸入空間中**密集抽樣**(dense sampling)，並用抽樣出的資料來訓練模型。
- 有了足夠多的參考樣本，模型在遇見全新樣本時，就能夠在過去的訓練資料中進行內插，而不是靠常識、抽象思考、或是參考外部知識來尋找答案。


當無法取得更多資料時，可以降低模型所能容納的資訊，或是在模型的擬合曲線上加入一些限制。

---
## Validation set

#### 1. Simple holdout validation
```python=
num_validation_samples = 10000
np.random.shuffle(data) # 打亂樣本順序
validation_data = data[:num_validation_samples]
training_data = data[num_validation_samples:]
```
:::warning
最簡單的評估方法，但是如果可用的資料很少，那麼validation data 和 testing data樣本也會很少。
:::

++**如何檢查資料是否太少?**++
先打亂資料順序，然後取出驗證的樣本，若每次重新洗牌後所訓練出的模型表現差異很大，那麼多半表示手上的資料太少了。

#### 2. K-fold validation

- 使用k-fold 驗證法之前，要先將資料拆分為相同大小的k個區塊。
- 在經過k次的選曲和訓練之後，取每一次分數的平均值為最終分數，然後參照此分數來調整模型的超參數。

```python!=
k = 3
num_validation_samples = len(data) // k
np.random.shuffle(data)
validation_scores = []
for fold in range(k):
    validation_data = data[num_validation_samples * fold: num_validation_samples * (fold + 1)] 
    training_data = np.concatenate(
        data[:num_validation_samples * fold],
        data[num_validation_samples * (fold + 1):])
    model.get_model() # 建立一個全新未訓練過的模型
    model.fit(training_data,...)
    
    validation_score = model.evaluate(validation_data, ...)
    validation_scores.append(validation_score)

validation_score = np.average(validation_scores) #最後驗證分數是每一折驗證分數的平均

"""重複以上程式並依照最終驗證分數來調整模型，之後進行以下程式:"""
model = get_model() #重建模型(其超參數已人工調整到最好)
model.fit(data, ...) #用所有非testing data 來訓練
test_score = model.evaluate(test_data) # 使用testing data 做最後的評估
```

#### 3. Iterated K-fold validation with shuffling
多次應用k-fold validation，並在每次分割k個區塊前均重新對資料洗牌，而最終驗證分數則是所有驗證分數的平均值。

:::warning
假設做了P次洗牌，則是取 P x K 次分數的平均值 (做 P 次 K-fold)，因此運算成本也會高很多。
:::

---
## 模型評估時的注意事項
#### 1. 資料代表性(data representativeness):
訓練資料集和測試資料集都有一定的代表性，足以反映資料的分布。因此在將資料拆分為訓練集和測試集之前，通常需要對資料做隨機洗牌(randomly shuffle)。

#### 2. 時間的方向性(the arrow of time):
如果我們試圖從過去的資料中，預測未來的狀態(天氣、股價)，
那麼就不該在分割資料集前做隨機洗牌，這樣會造成時間漏失(temporal leak)。此外，在進行具時間性的預測時，應確保測試資料的發生時間是在訓練資料之後。

#### 3. 資料中的重複性(redundancy in data):
如果資料中的某些資料點重複出現，然後進行分割與隨機洗牌後，可能導致重複的資料點出現在訓練集與測試集裡。這樣會造成使用相同的資料進行訓練與驗證，導致模型的表現不可靠。因此必須確保訓練集與測試集之間沒有交集。

---

## 提升模型的擬合表現

>想要達到完美擬合，勢必要先經過overfitting。我們無法預先知道邊界在哪，只有先越界了才知道。因此，我們處理任何機器學習問題時的初始目標，就是訓練出一個能展現基本的普適化能力，而且會發生overfitting 的模型。有了這樣的模型，我們才會開始專注在解決overfitting上，進而提升generalization能力。

### 訓練時通常會遇到這3個問題:
+ 訓練沒有成效，loss 降不下來
+ 成效尚可，但是沒普適化能力，甚至沒超越baseline
+ training loss 和 validation loss 都隨時間下降，表現也比baseline 好，但是無法達到過度配適，這也就代表模型還處於 underfitting 的階段。

### 解決分法:
#### 1. 調整gradient descent 的關鍵參數:
當loss 降不下來時，幾乎可以斷定是GD的參數配置出了問題:
- 優化器的選擇,
- 權重的初始值分布,
- **learning rate**,
- **batch size**

++**通常調整這兩個就好:**++
1. 降低或調高 learning rate
2. 增加 batch size
    - 若一個批次中有更多的樣本，就能提供更多資訊，同時雜訊也比較少。

#### 2. 利用既有的架構:
模型可以擬合訓練資料了，但是驗證準確率卻無法提升，這說明模型有在學習，但是無法普適化。

++**原因:**++
- 可能訓練樣本中的資訊不足以預測目標值
- 可能所使用的模型不適合來處理手上的問題
    - 例如: 用 fully connected layers 來處理**時間序列**預測問題.
    - 應該用 RNN 來處理

因此，為特定問題選擇合適的模型架構，對實現普適化來說是必要的。

:::info
在處理問題之前都應該先看看之前的成功案例，因為你通常不是第一個想要解決此問題的人。
:::

#### 3. 提升模型的 capacity:
如果我們的模型可以擬合資料，而且驗證損失也下降，似乎表示模型具有某程度的普適化能力。現在需要讓模型開始過度配適。

請記得，一定有辦法讓模型過度配適。如果無論如何都不會過度配適，那就可能是模型的**表徵能力**(representational power)不足。

我們可以透過:
- 增加神經層數量(make it deeper)、
- 使用更大的神經層(有更多神經元)、
- 或選擇更適合當下問題的神經層類形(選用更好的模型架構)

來提升表徵能力。



## 提升普適化能力
當模型具備一定普適化能力，而且也開始過度配適時，就可以開始專注在提升普適化能力了。

#### 1. 資料集篩選
深度學習的普適化能力源自資料本身的潛在結構。如果我們能在樣本間平滑地內插，那就有機會訓練出具備普適化能力的深度學習模型。

在收集資料上投入更多時間所得到的投資報酬率，通常比投資在發開更好的模型來的高。

++**幾個重點:**++
1. 確保手上有足夠的資料
2. 減少標註上的錯誤
3. 清理資料並處裡缺失值
4. 如果有很多特徵，但不知道哪些是有用的，應該先做特徵選擇。

#### 2. 特徵工程
特徵工程是指在訓練模型前，透過自身對手邊資料和機器學習演算法的理解，直接以人工的方式去轉換資料。

為了讓模型運作更順利，資料應該以更適合模型處理的形式來呈現。

:::success
**特徵工程的本質**:
透過更簡單的方法來表示問題，讓問題更容易被處理。同時，也讓潛在流形更平滑、更簡單、更有組織性。
:::

深度學習減少了對大多數特徵工程的需求，因為神經網路可以從原始資料中自動萃取有用的特徵，但是這不代表使用深度神經網路就不需要用到特徵工程，原因如下:
1. 良好的特徵可以在使用更少資源的狀況下，更有效的解決問題。
2. 良好的特徵讓我們能用更少的資料解決問題。如果樣本量少，則其特徵中的資訊就變得至關重要了。
#### 3. Early Stopping
深度學習模型永遠不會完全擬合訓練資料(若完全擬合，得到的模型就不具備任何普適化能力)。我們總是會在達到最小訓練損失之前就停止訓練。

最佳普適化的時間點是在underfitting 和 overfitting之間的平衡點。

在Keras中的典型作法，就是使用EarlyStopping (callback): 在訓練過程中，一但callback 程式發現驗證指標不再繼續提升，就停止訓練，並把最佳的模型state儲存下來。

#### 4. 將模型常規化
常規化技巧可以避免模型過度擬合訓練資料。

++**常見的常規化技巧:**++
1. 縮減神經網路的規模:
    - 很小的模型不會過度配適
    - 減縮模型的大小，及減少模型可用來學習的參數數量
    - 必須在too much capacity 和 not enough capacity之間取得平衡
    - 必需評估不同模型架構的表現(用驗證集評估)，以便找到正確的模型大小
    - 通常會從比較少的層數和參數開始，再逐漸增加層的大小或增加新的層，直到驗證損失不再進步為止。

:::warning
容量過大的模型的訓練損失通常很快就降至接近零，這是因為模型的容量越大，對訓練資料的學習速度就越快，但是過度配適的可能性也就越大，導致training_loss 和 validation_loss 有很大的差異。
:::

2. 加入權重常規化(weight regularization):
    -    給定一組訓練資料和一個神經網絡架構，通常簡單模型會比複雜模型更不容易過度配適。
    - 簡單模型: 參數值分配的熵比小的模型
    - 緩解過度配適的常用方法就是採用較小的權重值以限制模型的複雜性，進而讓權重值得分布更常規化。
    - 透過對損失函式中較大的權重加上代價(cost)來實現
        #### L1 regularization:
        所加入的cost 和權重的絕對值成正比
        #### L2 regularization:
        所加入的cost 和權重的平方成正比。也稱為 weight decay (權重衰減)

:::info
在Keras在做L1, L2 常規化時， cost 這一項只有在訓練時會使用，然後在驗證時(包含在測試時)會自動拿掉。
:::

```python=
"""將L2常規化加入模型中"""
from tensorflow.keras import regularizers

model = keras.Sequential([
    layers.Dense(units=16,    
            kernel_regularizer=regularizers.l2(0.002),                             
            activation='relu'),
    layers.Dense(16,
            kernel_regularizer=regularizers.l2(0.002),
            activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=["accuracy"])
history_l2_reg = model.fit(
                train_data, train_labels,
                epochs=20, batch_size=512, 
                validation_split=0.4)
```
L2(**0.002**) 表示該層的權重矩陣中，每個權重值都會加上[(0.002) x 權重值的平方] 到模型的總損失上 

有L2常規化的模型會比原始模型更能抵抗過度配適。

```python=
"""Keras 不同的常規化物件"""
from tensorflow.keras import regularizers

l1_reg = regularizers.l1(0.001)
l1_l2_reg = regularizers.l1_l2(l1=0.001, l2=0.002)
```
3. 加入丟棄法(Dropout)
    - 由於大型深度學習模型會有過度參數化的現象，因此在權重上強加常規化限制，對其模型容量和普適化並沒有太大的影響。
    - 對大型模型來說，需要用到丟棄法(Dropout)，主要是**在訓練期間隨機丟棄**神經網路層的一些輸出特徵。
    - 通常丟棄率介於0.2~0.5之間。
:::warning
在測試階段，並不會丟棄任何的特徵。取而代之的，是層的輸出值**將依照丟棄率的比例縮小**，以平衡訓練時特定輸出被歸零的影響(讓總數值不會偏差太大)。
:::

```python=
"""將Dropout添加到model"""
model = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
```

---
## Summary 機器學習的基礎
#### 1. 機器學習模型的目標就是普適化(generalization):即能夠正確判斷從未見過的輸入資料。
#### 2. 深層神經網路的普適化能力來自於:模型成功學會如何在訓練樣本之間進行內插(interpolate)，這相當於模型搞懂了訓練樣本的潛在流形(latent manifold)。這也說明了為何深度學習模型只能處理與訓練樣本十分接近的輸入。
#### 3. 深度學習的根本問題在於優化(optimization)與普適化(generalization)之間的平衡。想要實現普適化，模型必需得擬合訓練資料，不過擬合到一定程度後，卻會降低模型的普適化能力。
#### 4. 開發模型時，有必要找出方法來準確評估模型的普適化能力。
#### 5. 的首個目標就是讓其具備一定的普適化能力，而且有能力發生過度配適。我們可以透過這些方法來達成目標:
    -    調整learning rate
    -    調整batch size
    -    挑選合適的模型架構
    -    增加模型capacity
    -    或增加訓練時間
#### 6. 當模型開始過度配適時，我們的目標變成透過model regularization 來提升普適化能力。一般來說更大或品時更好的資料集會是提升模型普適化能力的首選，也可以用這些方法:
    -    減少model capacity
    -    加入dropout layer
    -    weight regularization
    -    或是early stopping


## 機器學習工作流程
### 1. 定義任務(Define the task)
了解客戶需求背後的問題領域與商業邏輯、收集資料、瞭解資料內容，必選擇衡量任務成功與否的標準。
#### 1-1 定義問題範圍
若想定義問題的範圍，通常需要和相關人員進行多次詳細的討論。
- 輸入資料是甚麼?
- 面對的是甚麼樣的機器學習任務?
- 現有的解決方法是甚麼?
- 是否需要考量特殊限制?
:::info
一定要先搞清楚整體的脈絡，才能把事情做好。
:::

#### 1-2 建立資料集
機器學習只能用來記憶訓練資料中的pattern，因此只能辨識曾經看過的東西。
- **投資在資料標註工具/方法**
    - 資料標註者是需要特地領域的專家，還是誰都可以做?
    - 需要特定知識才能標註資料嗎? 有可能訓練別人來標註嗎? 如果需要，該如何找相關專家?
    - 需要自行開發工具標註嗎?
- **留意不具代表性的資料**
    - 訓練資料一定要足以代表實際運作的資料(production data)
    - 如果無法使用實際運作的資料來訓練，一定要搞清楚訓練資料和實際運作的資料間的差異，然後主動去修正這些差異。
    - concept drift: 概念飄移，根源來自於實際資料的特性不斷變動，導致模型準確度下降。例如2014年訓練出來的音樂推薦引勤，放到現在可能已經不具參考性，準確率應該非常低。
    - 若想改善concept drift的問題，則需要持續收集資料、進行標註，並重新訓練模型。


#### 1-3 理解資料
在開始訓練模型之前，我們應該先探索和視覺化資料，以對資料有整體的概念，並思考它們如何協助實現預測能力，這樣做也有助於找出潛在問題。

- 檢查是否存在目標值洩漏(target leaking)的問題

#### 1-4 選擇測量成效的方法

要取得成功，必須先定義何謂成功。
成功可能是看:
- accuracy
- precision
- recall
- 又或是顧客回流率

成功的評量指標 (metrics) 會影響專案中的所有技術選擇 

---
### 2. 開發模型(Develop a model)
首先要準備好模型可處理的資料、並選擇評估模型的機制和要打敗的baseline。接著訓練第一個模型，這個模型要具備一定普適化能力，並且能夠過度配適。然後再進行常規化和調整模型，直到可以展現出最佳普適化的表現。
                                                                                            
#### 2-1 準備資料
- 向量化 (data vectorization) : 把資料轉換成張量
- 數值正規化 (normalization): 使每個特徵的標準差std 為 1, 平均值為 0
    ```python=
    """正規化"""
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    ```
- 處理缺失值
    - 可以選擇忽略缺失值
    - 如果這個特徵是分類，可以為該特徵新增一個分類值，用來代表此缺失值
    - 如果這個特徵是數值，應該避免隨意用一個數字來代表。這可能會造成潛在空間中出現不連續性，導致訓練出的模型難具有普適化能力。應該以該特徵的平均數或是中位數來代替。
    - 也可以訓練另一個模型來預測該缺失值的特徵值。

:::warning
如果已經知道測試集中有缺失值，但是訓練集中沒有缺失值，則神經網路將無法學會忽略缺失值。我們在這種情況下應該人工為訓練樣本製造一些缺失值，做法是多次複製數個訓練樣本，並從中刪除對應的特徵值。
:::
#### 2-2 選擇驗證機制
++**為什麼要選擇驗證機制?**++
模型的最終目標是要實現普適化，而整個模型開發中，每個決定都是由**驗證評量指標 (validation metrics)** 來引導，因此這個驗證機制可以用來衡量普適化的成效。

驗證機制有:
- holdout validtion set
- k-fold cross validation
- iterated k-fold validation

:::warning
一定要時刻留意驗證集的代表性 (representativity)，而且別讓訓練集和驗證集中出現重複的樣本。
:::

#### 2-3 超越baseline
第一個目標: 開發一個可以超越baseline 的小模型。

++**在這階段，要專注在3件事情上:**++
- **特徵工程**: 過濾掉不含有用資訊的特徵，也就是做 feature selection，然後根據自己對於問題的理解，找出可能有用的新特徵
- **選擇合適的既有架構**: 該使用何種架構? fully connected? convolutional? ..etc 或者深度學習適解決當前任務的最好辦法嗎? 還是該試試別的方式?
- **選擇足夠好的訓練配置**: 該選擇甚麼損失函數? 批次量? 學習率? 
---
| 問題類型 | 輸出層激活函數 | 損失函數 |
|:------ |:----------- | :-----------|
| 二元分類   | sigmoid | binary_crossentropy |
| 多類別、單標籤分類 | softmax| categorical_crossentropy |
| 多類別、多標籤分類    | sigmoid | binary_crossentropy |


#### 2-4 擴大規模: 開發一個會過度配適的模型
一旦獲得超過baseline能力的表現，我們要試著讓模型過度配適，目的是來了解這個模型是否足夠強大。

:::info
機器學習是在優化和普適化之間做取捨。
理想的模型是位於低度配適和過度配適的交界處、模型太小與模型太大之間。
要找出這個邊界的位置，我們必須先超過它(達到過度配適)。
:::
++**使模型過度配適的方法**++:
- 添加更多的神經層 (deeper)
- 讓每一層神經層更寬 (more neurons per layer)
- 訓練更多週期 (more epochs)

#### 2-5 將模型常規化並調整超參數
當模型可以超越baseline，而且有能力過度配適後，下一個目標就是++最大化模型的普適化能力++。
:::warning
這一步會佔大量時間: 反覆修改模型、反覆訓練
:::
**以下是該嘗試的方法:**
- 嘗試不同的架構: 添加或刪除神經層
- 使用dropout
- 如果模型不大，可使用L1, L2, 或L1_L2
- 嘗試不同的超參數
- 嘗試使用資料篩選 (data curation) 或 feature engineering: 收集和標註更多資料、找出更好的新特徵，或刪除似乎沒有用(無法提供有效資訊)的特徵。

:::danger
每次使用驗證集的回饋來調整模型時，都會有 information leak，將與驗證集有關的資訊洩漏到模型中，最終導致模型過度配適驗證資料。
:::

一旦找出令人滿意的模型配置，就可以重新用++訓練集++和++驗證集++來訓練最終的成品模型，並用測試集做最後一次評估。

如果測試集上的表現明顯差於驗證集上的表現，則可能表示驗證過程有問題，或者是在調整參數的過程中，模型開始對驗證集產生過度配適。

如果發上這種狀況，可以切換到可靠的評估驗證機制，例如: iterated k-fold validation

---
### 3. 部署模型(Deploy the model)
將模型部署到網路伺服器、行動app、等等，並監控模型的真實表現，然後開始收集建構下一代模型所需的資料。
 
#### 3-1 向客戶說明成果，並建立合理的期待
++**幾個重點:**++
- 清楚地說明模型能夠輸出甚麼結果
- 要將模型表現明確地與商務目標連接起來
- 和客戶及相關人士確認重要參數
- 涉及取捨的決策，都要和真正了解商業脈絡的專業人士來討論決定

#### 3-2 交付推論模型
- 以REST API部署模型
    - Flask
    - Tensorflow Serving
    - Google Cloud AI Platform, GCS
- 在裝置上部署模型
    - Tensorflow Lite
- 在瀏覽器上部署模型

在將模型匯入到 Tensorflow.js 或 Tensorflow Lite之前，都應該先進行優化

++**優化技巧:**++
- 權重剪枝(weight pruning): 減少模型中的參數量，只留下重要的那些
- 權重量化(weight quantization): 在訓練時，模型的權重值是單精準度浮點數(single-precision floating-point, float32)。不過在進行推論時，**可以將權重量化成8位元整數 (int8)**, 這樣可以將模型規模縮小至原先的1/4，但準確度仍保持在接近原本的水準。


#### 3-3 監控模型的運作
這邊可以考慮引入 A/B 測試

#### 3-4 維護模型
概念飄移 (concept drift) 會導致模型隨時間失去準確度。

一旦模型正式啟用，就該準備下一代模型了。

++**因此我們需要:**++
- 時刻關注實際運作的資料中的變動，是否出現新特徵?是否該擴充? 或是修正標籤?
- 持續收集與標註資料，並改進標註過程
- 專注在收集那些現有模型很難分辨的樣本，這些才能幫助優化

## Summary 機器學習的工作流程
#### 1. 開始新的機器學習專案前，一定要釐清:
    -    最終目標是甚麼? 有那些限制?
    -    收集與標註資料集; 確保已經深入了解這些資料的本質
    -    要如何評估結果是否成功? 使用甚麼評估指標(metrics)來監控模型在驗證集上的表現?
#### 2. 當我們了解問題，也收集到合適的資料集後，便可以開始著手開發模型:
    -    準備資料
    -    確認驗證機制: k-fold? simple holdout? 要取用資料集的哪個區塊做驗證?
    -    實現統計能力: 打敗baseline
    -    擴大規模: 開發能更過度配適(overfit) 的模型
    -    根據模型在驗證集上的表現，對模型進行常規化並調整超參數
    -    找到最理想的配置後，用訓練集和驗證集做最終訓練，並用測試集做最後驗證
#### 3. 當模型在測試集上的表現不錯時，就可以進入部署階段:
    -    首先，確認客戶的期待是合理適當的
    -    優化用來進行推論的最終模型，並根據所需選擇部署環境
    -    投入生產運作後，持續監控並收集與標註新資料，以便開發下一代的模型

---

## [Ch.7 深入探討 Keras](https://colab.research.google.com/drive/1fGylrZv4uPI1R7cfkJmFELZB403-_c7U#scrollTo=Z71VCpP-JSkR)


## Summary 深入探討 Keras
#### 1. 根據 "**逐步提升複雜度**" 的基本原則 Keras 提供了一系列的工作流程。
#### 2. 建構模型可以透過:
    -    Sequential類別
    -    Funtional API 
    -    Model Subclassing
:::info
最常用到的是 Functional API
:::
#### 3. 若想訓練或評估模型，最簡單的方法就是呼叫fit() & evaluate()。
#### 4. Keras的內部callbacks讓我們能夠在呼叫fit()後監控模型的內部狀態，並根據模型狀態自動決定該採取的行動。
#### 5. 也可以透過手動改寫 Model子類別的 train_step()方式來控制fit()的行為。
:::info
在自定義的 Subclassing Model裡override訓練函式，之後讓fit()來使用。這樣既可以有自己的訓練函式，又可以用上fit()內建的便利功能，例如: callbacks。
:::
#### 6. 若想要更進一步改變fit()的行為，也可以從零開始，設計自己的訓練迴圈。這對於要自行實作全新訓練演算法的研究者來說，是非常有用的功能。


---
## ch.8 電腦視覺的深度學習簡介
:::info
要縮小對特徵圖的採樣數，比較設定步長，一般來說更傾向使用 max-pooling.
:::

++**max-pooling**:++
往往比 average-pooling 或調整stride 效果來得好，原因是：特徵通常是源自於空間中某些有特色的pattern, 因此要取其最大值才更具訊息性，如果取他們的平均值，那特色就被淡化了。如果把stride 調大，大步滑動filter，可能會錯過寶貴的訊息。

因此最合理的採樣策略是先用 stride = 1來密集掃描特徵圖，然後在採樣的小區塊上查看特徵的最大值(max-pooling)。

### 少量資料集的訓練
3種技術來減少過度配適和提升準確度:
1. data augmentation
2. feature extraction with a pre-trained model
3. fine tune a pre-trained model

**本質上深度學習模型是可高度再利用的**

以電腦視覺而言，許多預先訓練好的模型（通常是ImageNet資料集進行訓練) 都是可以公開下載。以這些預先訓練好的模型為基礎，再加以少量資料的補強訓練，就能產出另一個強大的電腦視覺模型。

### padding='same', 'valid', 'full'

https://zhuanlan.zhihu.com/p/62760780

- padding= 'full':
    -  從filter的邊角 和 image 的邊角相交開始就做卷積
- padding= 'same': 
    - 當filter 的中心格與image的邊角重合時，開始做卷積運算．
    - 在forward pass讓特徵圖大小保持不變
- padding= 'valid':
    - 當filter 全部都在image裡時，才做卷積 


### tf.data.Dataset
- 提供非同步資料擷取
- Dataset物件每次只產生單一樣本

:::success
**Dataset**
- 可透過.batch()方法來變更批次量
- .shuffle(buffer_size) 來打亂元素順序
- .prefetch(buffer_size) 固定預先取得 buffer_size個樣本到GPU, 以更好的利用硬體資源
- map(callable): 對資料集中的每個元素進行特定轉換, callable為某一函式, 預期輸入參數為資料集的單一元素，傳回值則為轉換後的資料
:::


```python=
"""只有在當前週期的val_loss 比上一次的val_loss 都來的低時，才會儲存最新的模型"""
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='convnet_from_scratch.keras',
        save_best_only=True,
        monitor='val_loss'
    )
]
```
---
### Data Augmentation
:::warning
減緩overfitting，專門針對電腦視覺，且在深度學習模型處理影像時都會用到
資料擴增法 data augmentation
:::

```python=
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"), ### 隨機將50%的輸入影像水平翻轉
    layers.RandomRotation(0.1), ### 旋轉輸入影像,幅度為[-10%, +10%]範圍內的一隨機值
    layers.RandomZoom(0.2) ### 放大或縮小影像, 幅度為[-20%, +20%]內的一隨機百分比
])
```

不過光是 data augmentation 還不夠完全擺脫過度配適，因為 data augmentation 是從現有的訓練資料集內重新混合資料，所以影像間仍有相關。

為了近一步防止過度配適，必須加入 Dropout。

:::danger
和 Dropout 一樣，data augmentation layer 在推論(inference)階段: predict(), evaluate() 會自動變成無作用。
:::

---
### Using Pre-trained models
如果原資料集足夠大而且具通用性，那麼pre-trained model 訓練出的模型其空間層次特徵(spatial hierarchy features)就足以充當是學世界的通用模型。

例如用ImageNet先訓練出一個神經網路(其辨識項目主要是動物和日常用品)，然後重新訓練這個已訓練完成的網路，去辨識和原本樣本天差地遠的家具產品。

深度學習的優勢在於 ++**學習到的特徵可移植到不同問題上**++，使得深度學習對於樣本資料量較少的場合也非常有效。

:::info
ImageNet 資料集包含140萬個標註好的影像和1000個不同類別。
ImageNet包含許多動物類別(包含貓和狗)。
:::

**使用pre-trained model有兩種方法:**
- feature extraction
- fine-tuning


#### Feature extraction
**用於影像分類的cnn 包含兩部分:**
1. cnn base
2. classifier

Feature extraction的做法是以一個訓練模型的 cnn base 來處理新資料，並以其輸出結果來訓練新的分類氣。


*為什麼只用 cnn base 而不包含 classifier?*

*為什麼只用 cnn base 而不包含 classifier?*:::info
cnn 的特徵圖代表某張影像的通用概念圖，因此無論面臨何種電腦視覺問題，都可能是有用的。

但是 classifier學習到的表示法可能只適用於目前模型所要分類的類別，僅包含整個影像中相關類別的出現機率的資訊。

另外 cnn base 輸出的特徵圖仍會包含物件出現的位置訊息，但 classifier 或密集層學習到的表示法則不包含物件在輸入影像中的任何位置訊息，因此，對於需要考量物件位置的問題來說，dense layer 產生的特徵絕大多數是沒用的。
:::

#### Fine-tuning
1. 在已經訓練過的基礎神經網路上增加自定義的神經網路(分類器)
2. freeze base
3. 訓練新增的自定義的分類器
4. unfreeze某幾層(**但是不應該unfreeze BatchNormalization Layer**)
5. 共同訓練unfreeze的層和分類器

base 中的低層是對更通用、可重複使用的特徵進行編碼，而更高層則是對更特定的特徵進行編碼。

++**為什麼不微調整個 base:**++
1. 對更特定的特徵進行微調將更有效果，因為這些特徵需要重新調整以用於新問題 (低層的現象很通用，所以已經適用於新問題了)，而微調低層會出現效果下降的現象。

2. 訓練的參數越多，越有可能發生過度配適。而且這麼做的話在少量資料上訓練會有風險。


## Summary 電腦視覺的深度學習簡介
#### 1. convnets 最適合來處理電腦視覺的任務，即使只有非常小的資料集，也能重頭訓練出一個表現不錯的模型．
#### 2. 使用小規模資料集來訓練時，過度配適會是主要問題．而 data augmentation 是對抗過度配適的最強大工具．
#### 3. 透過 feature extraction 可以將既有的cnn網路應用在新的資料集中．在處理小規模的影像資料集時，這個技巧特別有用．
#### 4. 為了提升 feature extraction 的效果，可以使用 fine tuning，將模型先前已經學習到的部分表示能力應用在新任務上

---
## Ch.9 Advanced Computer Vision

在做影像分割時，我們用strides = 2 來做 Downsampling，而不是用MaxPooling2D()。

原因在於，在做影像分割時，我們必須為每個像素來生成目標遮罩，以此做為模型的輸出，所以會很注重資訊在圖片中的空間位置(spatial location)。

而當我們用MaxPooling2D時，會完全破壞每個池化窗格的位置資訊: 因為只會回傳最大值，我們無法知道該值來自窗格中四個位置的哪一個。

因此，雖然MaxPooling2D在分類中表現很好，但是在分割任務中卻會帶來壞處。而 strided convolution (strides > 1) 在image segmentation任務中，可讓我們在更能保留位置資訊下，對特徵圖進行downsampling。


### Conv2DTranspose
```python=
import tensorflow as tf

x = tf.random.normal(shape=(1, 100, 100, 64))

conv = layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')
trans_conv = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')

y = conv(x)
print(f"shape of output of Conv2D: {y.shape}")


z = trans_conv(y)
print(f"shape of output of Conv2DTranspose: {z.shape}")

```

> #### output:
> shape of output of Conv2D: (1, 50, 50, 128)
> shape of output of Conv2DTranspose: (1, 100, 100, 64)

:::info
一般來說, 由較小層（神經單元數較小）組成的較深堆疊，表現會比較大層組成的淺堆疊更好。
:::

---
### Residual connection
把輸入添加到區塊的輸出中，代表輸出的 shape 應該和輸入相同．

如果區塊中包含過濾器數量逐漸增加的卷積層，或最大池話層，就無法滿足 shape 相同的條件．

在這種情況下，應該使用沒有activation 的 kernel_size=1x1 Conv2D層來將residual 線性投影成區塊輸出的shape, 通常會使用 padding='same'來避免因邊界效應導致的空間subsampling.

如果使用MaxPooling2D就要加一個strides=2 的 1x1 Conv2D卷積，
若不使用MaxPooling2D, 就只有在過濾器數量發生變化時才進行殘差


### Batch Normalization

- 用BN層時不需要加入bias, 我們可以透過use_bias = False 來創建沒有偏值的層，也看起來更精簡

```python=
"""不建議的 bn 用法"""
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
x = layers.BatchNormalization()(x)
```

```python=
"""建議的 bn 用法"""
x = layers.Conv2D(filters=32, kernel_size=3, use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x) ### 正規化後，再使用activation

```

這種做法的直觀原因是，bn 會將輸入置中於零，而relu 會把零當成保留或丟棄激活資料的判斷關鍵，因此在激活前做正規化，可以最大限度地運用relu

:::warning
也可以先做conv -> activation -> bn 
模型結果不一定比較差
:::

### 目前為止學過的 cnn 原則:
- 模型應該由重複的區塊組成，通常包括多個卷積層和一個最大池化層
- 層中的channels （過濾器數量）應該逐漸增加，而特徵圖的尺寸逐漸變小
- 深又窄的神經網路，比淺又寬的好
- 在區塊旁邊使用殘差連接，有助於訓練更深的網路
- 在 convolution層之後使用 BN 是有利的
- 使用 separableConv2D 來取代 Conv2D 是有利的，因為其參數使用率較高

## LSTM
:::info
LSTM單元的功能: 允許在未來重新使用過去的資料，從而對抗梯度消失的問題。
:::

---

## GAN
實作技巧:
1. 使用strides =2 來降採樣鑑別器的特徵圖
2. 使用常態分佈從潛在空間中取樣點，而不是使用均勻分佈
3. 引入隨機性:
    -    使用Dropout
    -    在discriminator的資料標籤中增加隨機雜訊
4. 不用MaxPooling2D, 用strides=2
5. 建議使用LeakyReLU來漸緩稀疏性的發生
6. 在使用strided Conv2D 或是 Conv2DTranspose，使用可以被步長大小整除的kernel_size
    - 例如: strides=2, kernel_size=(4,4)


### GAN = Discriminator + Generator

1. 在每個訓練週期中, 會執行以下操作:
2. 在潛在空間中隨機取樣一些點（隨機雜訊））
3. 將這些點輸入生成器, 以生成一批假影像
4. 將這批假影像和真實影像混再一起, 另外再準備對應的標籤，其中包括 “真的” 和 “假””
5. 使用這些混合的影像和標籤來訓練 discriminator, 先訓練 discriminator
6. 在潛在空間中重新取樣一批新的點(雜訊)
7. 將這些點輸入 generator 以生成一批假映像，然後準備對應的標籤，但這次把這些假影像打上“真標籤“的標籤
8. 將這些假影像和標籤輸入 discriminator 進行預測．並將預測結果與對應的標籤相比較，進而得出一個損失。
9. 朝著減少損失的方向更新 generator 的權重，以便讓 discriminator 將假影像預測為真實影像．

