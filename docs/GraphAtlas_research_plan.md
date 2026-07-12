# GraphAtlas：基于可学习黎曼图册的坐标无关图表示学习

## 暂定标题

**GraphAtlas: Coordinate-Invariant Message Passing over Learned Riemannian Atlases**

备选：

**Beyond Geometry Mixtures: Learning Coordinate-Compatible Local Atlases for Heterogeneous Graphs**

不建议继续在标题中突出 mixed geometry 或 curvature routing。这两个方向已经比较拥挤。

---

# 一、核心论点

现有几何图学习通常研究：

> 一个节点应该被嵌入哪个几何空间？

GraphAtlas 研究一个更基础的问题：

> 当不同局部区域采用不同坐标表示时，图消息应当如何在这些坐标系统之间正确传递，并保证最终预测不依赖任意的坐标选择？

这一区别非常重要。

GraphMoRE 已经通过局部拓扑编码，为节点选择和融合不同曲率的黎曼专家，并使用跨空间对齐计算节点距离。GeoMoE 又加入了曲率引导路由。ARGNN 进一步学习节点级连续、各向异性的黎曼度量。继续做多空间选择、连续曲率或节点级几何分配，已经很难构成根本创新。

GraphAtlas 不再把几何空间当作若干专家，而是将图表示定义为：

[
\mathcal A=
\left{
U_k,\phi_k,\psi_k,g_k,\tau_{lk}
\right}_{k=1}^{K},
]

其中：

* (U_k) 是相互重叠的局部节点区域；
* (\phi_k) 是局部坐标编码器；
* (\psi_k) 是局部坐标解码器；
* (g_k) 是局部坐标中的黎曼度量；
* (\tau_{lk}) 是重叠区域上的坐标转换。

模型不是先在多个空间中独立编码，再进行加权融合。

模型直接在局部坐标中执行消息传播。来自其他区域的消息必须先经过坐标转换，才能参与当前区域的聚合。

整篇论文的核心句应当是：

> **GraphAtlas replaces expert fusion with coordinate-compatible transport.**

---

# 二、现有方法缺少什么

## 1. 几何 MoE 缺少坐标兼容性

GraphMoRE 和 GeoMoE 为节点分配多个几何专家，但不同专家的输出最终仍通过加权、拼接或共同切空间对齐进行融合。它们回答的是不同节点需要何种几何，却没有要求两个局部表示必须是同一个对象在不同坐标中的描述。

因此，普通几何 MoE 可以写成：

[
z_i=\sum_k q_{ik}z_i^{(k)}.
]

不同 (z_i^{(k)}) 可以拥有完全不同的语义。只要最终任务损失较低，模型不需要保证它们之间存在合法坐标转换。

GraphAtlas 要求：

[
z_i^{(l)}
\approx
\tau_{lk}\left(z_i^{(k)}\right),
\qquad
v_i\in U_k\cap U_l.
]

两个表示必须是同一个局部对象的不同坐标表达。

## 2. 节点级度量场仍依赖单一坐标系统

ARGNN 已经学习连续、各向异性的节点级黎曼度量，因此 GraphAtlas 不能把可学习局部度量本身作为主要创新。

GraphAtlas 与 ARGNN 的区别是：

* ARGNN 在一个共同特征坐标系统内，为不同节点学习不同度量；
* GraphAtlas 允许图由多个重叠坐标域表示；
* GraphAtlas 显式学习坐标转换；
* GraphAtlas 要求预测对局部坐标重参数化保持不变。

节点度量描述的是局部如何测量距离。

图册描述的是不同局部坐标如何组成同一个全局对象。

## 3. Sheaf 方法具有映射，但没有图册结构

Neural Sheaf Diffusion 为节点和边分配向量空间与限制映射。Connection-Laplacian 方法也会对相邻节点的切空间进行对齐。因此，单纯加入线性转换矩阵不能构成创新。

GraphAtlas 必须与 sheaf 方法明确区分：

| 方法           | 局部对象         | 映射粒度          | 映射来源      | 几何目标         |
| ------------ | ------------ | ------------- | --------- | ------------ |
| Neural Sheaf | 节点、边上的 stalk | 每条边           | 自由学习的限制映射 | 运输感知扩散       |
| GraphAtlas   | 少量重叠 chart   | chart overlap | 编码器和解码器诱导 | 图册一致性与局部度量保持 |

GraphAtlas 不为每条边学习独立矩阵，而是让同一对局部区域共享非线性坐标转换。转换必须满足重构、可逆、路径一致和度量兼容。

## 4. 普通 atlas learning 不解决图传播

多图册自动编码器已经可以从局部编码器和解码器构造转换映射，并由重构一致性得到 cocycle 条件。因此，“首次学习多个 chart 和 transition”本身也不能作为贡献。

GraphAtlas 的真正创新应当是：

> **首次将可学习图册用于坐标无关的图消息传播，并分析跨 chart 运输误差如何影响图表示失真与异配传播。**

---

# 三、问题定义

给定属性图：

[
G=(V,E,X),
]

GraphAtlas 学习一个稀疏重叠覆盖：

[
V=\bigcup_{k=1}^{K}U_k.
]

每个节点具有软成员度：

[
q_{ik}\in[0,1],
\qquad
\sum_kq_{ik}=1.
]

与 MoE 不同，(q_{ik}) 不表示选择第 (k) 个专家，而表示节点是否位于第 (k) 个局部坐标域中。

需要满足：

### 覆盖性

[
\max_k q_{ik}\geq \epsilon_{\mathrm{cover}}.
]

每个节点至少被一个 chart 有效覆盖。

### 稀疏性

[
|q_i|_0\ll K.
]

大部分节点只属于一到两个局部区域。

### 重叠性

对于局部区域边界节点，应当存在：

[
q_{ik}>0,\qquad q_{il}>0.
]

重叠不是门控不确定性，而是坐标转换存在的前提。

### 连通性

构造 chart overlap graph：

[
\mathcal G_{\mathcal A}
=======================

({1,\ldots,K},E_{\mathcal A}),
]

若 (U_k\cap U_l\neq\varnothing)，则存在 chart 边 ((k,l))。

整个图册应当通过 overlap graph 连通。

---

# 四、模型设计

## 1. 图原生的局部 chart 发现

为节点构造不依赖标签的局部签名：

[
s_i=
\left[
s_i^{\mathrm{struct}},
s_i^{\mathrm{spectral}},
s_i^{\mathrm{semantic}},
s_i^{\mathrm{relation}}
\right].
]

其中可使用：

* 多尺度度分布；
* 局部聚类与 motif 统计；
* 扩散返回概率；
* 局部谱响应；
* 邻居特征差异；
* 结构、属性和高阶关系的一致程度。

成员度通过稀疏映射得到：

[
q_i=\operatorname{entmax}(f_\theta(s_i)).
]

不建议直接将结构视图、属性视图和原型视图设置为三个 chart。

正确方式是让这些视图提供局部区域发现的证据。一个 chart 可以同时利用多种关系证据。

区域正则仅保留三项：

[
\mathcal L_{\mathrm{cover}},
\qquad
\mathcal L_{\mathrm{sparse}},
\qquad
\mathcal L_{\mathrm{balance}}.
]

不要加入过多区域损失，否则方法会显得依赖人工调参。

---

## 2. 局部坐标编码器和解码器

首先得到共享的观察表示：

[
h_i^{(0)}
=========

f_{\mathrm{obs}}(x_i,s_i).
]

这里的共享空间只是高维观察空间，不承担最终几何建模。

对于每个 chart：

[
u_i^{(k)}
=========

\phi_k(h_i^{(0)})
\in\mathbb R^d,
]

[
\widehat h_i^{(k)}
==================

\psi_k(u_i^{(k)}).
]

局部重构损失为：

[
\mathcal L_{\mathrm{rec}}
=========================

\sum_{i,k}
q_{ik}
\left|
\psi_k(\phi_k(h_i^{(0)}))-h_i^{(0)}
\right|_2^2.
]

每个 chart 的度量不再人为指定为欧氏、双曲或球面，而是通过解码器的 pullback metric 定义：

[
g_k(u)
======

J_{\psi_k}(u)^\top
J_{\psi_k}(u)
+
\epsilon I.
]

这一设计有三个好处：

1. (g_k) 自动保持正定；
2. 每个 chart 可以表达连续变化和各向异性的局部几何；
3. 坐标变换后，度量具有明确的变换规则。

欧氏、双曲和球面可以作为合成数据中的几何类型，或者作为度量正则的原型。

它们不再是三个并行专家。

---

## 3. 由 chart 自动编码器诱导转换

对于两个重叠区域 (U_k) 和 (U_l)，定义：

[
\tau_{lk}
=========

\phi_l\circ\psi_k.
]

即：

[
\tau_{lk}(u)
============

\phi_l(\psi_k(u)).
]

它先将 chart (k) 的坐标解码回观察空间，再编码到 chart (l)。

不要为每一对 chart 独立学习一个自由矩阵 (W_{kl})。

自由矩阵会导致两个问题：

* 容易退化为共同切空间对齐；
* 路径一致性可能通过矩阵分解被平凡满足。

由编码器和解码器诱导 transition，可以让转换与 chart 本身共享语义，并自然获得近似可逆性：

[
\tau_{kl}\circ\tau_{lk}
\approx I.
]

三重重叠区域还应满足：

[
\tau_{mk}
\approx
\tau_{ml}\circ\tau_{lk}.
]

对应的 cocycle defect 为：

[
\mathcal L_{\mathrm{coc}}
=========================

\sum_{i,k,l,m}
q_{ik}q_{il}q_{im}
\left|
\tau_{mk}(u_i^{(k)})
--------------------

\tau_{ml}
\left(
\tau_{lk}(u_i^{(k)})
\right)
\right|_2^2.
]

---

## 4. 度量兼容性

只约束转换后的坐标接近仍然不够。

真正的黎曼图册要求局部度量在转换下保持兼容。

若 (\tau_{lk}) 将 chart (k) 转换到 chart (l)，则应满足：

[
g_k(u)
\approx
J_{\tau_{lk}}(u)^\top
g_l(\tau_{lk}(u))
J_{\tau_{lk}}(u).
]

因此定义：

[
\mathcal L_{\mathrm{metric}}
============================

\sum_{i,k,l}
q_{ik}q_{il}
\left|
g_k(u_i^{(k)})
--------------

J_{\tau_{lk}}^\top
g_l(u_i^{(l)})
J_{\tau_{lk}}
\right|_F^2.
]

这项约束是 GraphAtlas 与普通 geometry MoE 拉开差距的关键。

普通 MoE 只要求多个表示对任务有用。

GraphAtlas 要求它们描述同一个局部几何对象。

---

## 5. Atlas Transport Convolution

这是整篇论文最重要的模型模块。

每个节点在 chart (k) 中维护局部隐藏向量：

[
a_i^{(k)}
\in
T_{u_i^{(k)}}U_k.
]

假设节点 (j) 的消息目前位于 chart (l)，节点 (i) 在 chart (k) 中进行聚合。

必须先使用 transition Jacobian 将消息变换到 chart (k) 的坐标基：

[
P_{l\rightarrow k}(u_j^{(l)})
=============================

J_{\tau_{kl}}(u_j^{(l)}).
]

跨 chart 消息为：

[
m_{j\rightarrow i}^{(k)}
========================

\sum_l
q_{jl}
\alpha_{ij}^{kl}
P_{l\rightarrow k}(u_j^{(l)})
a_j^{(l)}.
]

局部更新为：

[
a_i^{(k)\prime}
===============

\sigma
\left(
W_{\mathrm{self}}a_i^{(k)}
+
\sum_{j\in\mathcal N(i)}
m_{j\rightarrow i}^{(k)}
\right).
]

若节点 (i) 和 (j) 位于同一个 chart，则：

[
P_{k\rightarrow k}=I.
]

若两个 chart 没有直接重叠，则通过 overlap graph 上的最短 chart 路径组合转换：

[
P_{l\rightarrow k}
==================

P_{r_{H-1}\rightarrow k}
\cdots
P_{l\rightarrow r_1}.
]

最终将 chart 内的隐藏向量映射回共同观察空间：

[
r_i
===

\sum_k
q_{ik}
J_{\psi_k}(u_i^{(k)})
a_i^{(k)}.
]

任务预测为：

[
\widehat y_i=f_{\mathrm{task}}(r_i).
]

这里的加权发生在信息已经通过合法坐标转换并映射回观察空间之后。

它不再等价于直接融合多个专家输出。

---

# 五、论文必须证明的核心性质

理论部分只保留两条主线。

## 定理一：局部坐标重参数化不变性

对于任意 chart (k)，考虑光滑可逆的坐标变换：

[
\rho_k:\mathbb R^d\rightarrow\mathbb R^d.
]

重新定义：

[
\widetilde\phi_k
================

\rho_k\circ\phi_k,
]

[
\widetilde\psi_k
================

\psi_k\circ\rho_k^{-1}.
]

转换随之变为：

[
\widetilde\tau_{lk}
===================

\rho_l
\circ
\tau_{lk}
\circ
\rho_k^{-1}.
]

局部度量按照 pullback 规则变化。

需要证明，在 transition Jacobian、局部隐藏向量和度量均按正确规则变换时，Atlas Transport Convolution 的观察空间输出满足：

[
\widetilde r_i=r_i.
]

因此：

[
\widetilde{\widehat y}_i
========================

\widehat y_i.
]

意义是：

> 模型预测不依赖研究者为每个局部区域任意选择了什么坐标。

这是普通 MoE、全局嵌入和简单切空间对齐不具备的保证。

它也可以转化成非常有力的实验。

---

## 定理二：图册失真分解

定义 chart 内局部几何失真：

[
\epsilon_k
==========

\sup_{i,j\in U_k}
\left|
d_{g_k}(u_i^{(k)},u_j^{(k)})
----------------------------

d_G^{\mathrm{loc}}(i,j)
\right|.
]

定义转换误差：

[
\delta_{kl}
===========

\sup_{i\in U_k\cap U_l}
\left|
\tau_{lk}(u_i^{(k)})
--------------------

u_i^{(l)}
\right|.
]

定义 cocycle defect：

[
\eta
====

\sup_{k,l,m}
\left|
\tau_{mk}
---------

\tau_{ml}\circ\tau_{lk}
\right|.
]

若两个节点之间需要经过至多 (H) 次 chart 转换，则目标上界应具有如下形式：

[
D_{\mathrm{atlas}}
\leq
C_1\max_k\epsilon_k
+
C_2H\max_{k,l}\delta_{kl}
+
C_3H\eta.
]

这比简单写成局部失真加转换误差更严谨，因为跨 chart 路径会累积误差。

进一步构造一类混合局部增长图，使任意单一常曲率空间满足：

[
\inf_c D_{\mathrm{single}}(G,c)
\geq \alpha,
]

而存在一个有限 chart 图册满足：

[
D_{\mathrm{atlas}}
\leq\beta,
\qquad
\beta<\alpha.
]

不需要证明恢复真实流形。

只需要证明：

1. 固定全局几何存在严格失真下界；
2. atlas 可以取得更低失真；
3. transition 和 cocycle defect 决定跨区域误差。

---

# 六、为什么它与异配有关

不能只把 heterophily 当作实验标签。

需要明确指出，传统消息传播的问题不仅是邻居标签不同，还可能是邻居表示位于不兼容的局部坐标中。

普通传播直接计算：

[
h_i'
====

\sum_{j\in\mathcal N(i)}
W h_j.
]

它隐含假设所有邻居的隐藏特征具有相同的坐标含义。

在多局部关系区域中，同一维特征在两个区域内可能表示不同方向或不同局部角色。

GraphAtlas 先进行运输：

[
h_i'
====

\sum_{j\in\mathcal N(i)}
P_{j\rightarrow i}h_j.
]

论文可以将异配分为两类：

### 标签异配

相邻节点的类别不同。

### 坐标异配

相邻节点处于不同关系区域，其表示不能直接比较或相加。

GraphAtlas 主要解决第二类问题，并通过正确运输改善第一类异配下的消息传播。

这一表述比“异配区域需要不同曲率”更加具体，也更容易形成理论分析。

---

# 七、新合成基准：Atlas-Het

原来的树、SBM、环和二部图拼接还不够严谨。

新版基准必须同时具有：

* 真实 chart；
* 真实 transition；
* 真实局部距离；
* 可控 heterophily；
* 可控 overlap；
* 可控 coordinate intervention。

## 1. 几何生成

首先定义一个已知的低维潜在几何对象，并使用多个重叠参数化覆盖它。

可选择：

* 具有正、负和接近平坦曲率区域的嵌入曲面；
* 具有已知 atlas 的球面、环面或非定向曲面；
* 由多个局部参数域通过已知 diffeomorphism 拼接得到的空间。

在真实测地距离下生成基础边：

[
P(A_{ij}^{\mathrm{geo}}=1)
==========================

\sigma
\left(
a-bd_{\mathcal M}(x_i,x_j)
\right).
]

## 2. 异配关系生成

再通过关系兼容矩阵生成额外边：

[
P(A_{ij}^{\mathrm{rel}}=1\mid y_i,y_j)
======================================

C_{y_i y_j}.
]

这样可以分别控制：

* 几何结构；
* 标签兼容性；
* 跨区域边比例；
* 局部异配率；
* 结构噪声。

最终图为：

[
A=A^{\mathrm{geo}}\lor A^{\mathrm{rel}}.
]

## 3. 真实标注

基准提供：

* chart membership；
* overlap-node label；
* 真实 transition；
* transition Jacobian；
* 局部测地距离；
* 局部曲率或度量；
* 几何边和关系边来源；
* 区域边界。

## 4. 坐标干预测试

这是基准最有价值的部分。

对每个 chart 独立施加随机可逆坐标变换：

[
u^{(k)}
\mapsto
\rho_k(u^{(k)}).
]

可以包括：

* 随机正交变换；
* 非奇异仿射变换；
* 平滑非线性扭曲；
* 不同尺度和轴方向变化。

图结构和节点语义保持不变。

一个真正的图册模型应当保持预测基本稳定：

[
\operatorname{InvErr}
=====================

\frac{1}{|V|}
\sum_i
\left|
p_i-
\widetilde p_i
\right|_1.
]

这项实验可以直接检验模型是否真的学习了坐标无关表示。

普通 geometry MoE 即使准确率很高，也未必能通过该测试。

---

# 八、真实数据实验

## 主要数据

重点使用：

* Roman-empire；
* Amazon-ratings；
* Minesweeper；
* Tolokers；
* Questions；
* Actor；
* Chameleon；
* Squirrel。

补充同配和层级图：

* Cora；
* CiteSeer；
* PubMed；
* Amazon Photo；
* Airport 或其他层级图。

## 任务

不建议一次做太多任务。

主文保留：

1. 节点分类；
2. 链路预测；
3. Atlas-Het 几何恢复。

跨图归纳可以作为扩展实验，而不是主线任务。

## 必须比较的模型

### 全局几何

* Euclidean GNN；
* Hyperbolic GNN；
* Spherical GNN；
* Product manifold；
* 全局可学习曲率。

### 局部几何

* GraphMoRE；
* GeoMoE；
* ARGNN。

### 运输方法

* Neural Sheaf Diffusion；
* Connection-Laplacian SNN。

### GraphAtlas 消融

* Atlas without overlap；
* Atlas with free linear transitions；
* Atlas without transition transport；
* Atlas without metric compatibility；
* Atlas with post-hoc MoE fusion；
* Full GraphAtlas。

参数量和训练预算必须匹配。

---

# 九、关键诊断实验

## 1. 边界节点性能

分别报告：

[
\operatorname{Acc}*{\mathrm{interior}},
\qquad
\operatorname{Acc}*{\mathrm{boundary}}.
]

GraphAtlas 的优势必须主要出现在：

* overlap 节点；
* 跨 chart 边；
* 局部结构冲突区域。

如果只提高普通内部节点性能，说明收益可能来自模型容量。

## 2. 坐标干预稳定性

比较坐标变换前后的：

* 分类概率变化；
* 节点表示距离变化；
* 链路分数变化。

这应成为主表，而不是附录实验。

## 3. 路径一致性

对于 chart 路径 (k\rightarrow l\rightarrow m)，测量：

[
E_{\mathrm{path}}
=================

\left|
\tau_{mk}(u)
------------

\tau_{ml}(\tau_{lk}(u))
\right|.
]

同时比较不同路径运输同一消息后的差异。

## 4. 度量兼容性

测量：

[
E_{\mathrm{metric}}
===================

\left|
g_k-
\tau_{lk}^{*}g_l
\right|_F.
]

该指标比仅可视化曲率更有说服力。

## 5. 图册恢复

在 Atlas-Het 上报告：

* chart ARI；
* overlap-node F1；
* transition error；
* local distortion；
* cross-chart distortion；
* coordinate invariance error。

---

# 十、最终训练目标

训练目标控制在五部分以内：

[
\mathcal L
==========

\mathcal L_{\mathrm{task}}
+
\lambda_{\mathrm{rec}}
\mathcal L_{\mathrm{rec}}
+
\lambda_{\mathrm{geo}}
\mathcal L_{\mathrm{geo}}
+
\lambda_{\mathrm{atlas}}
\left(
\mathcal L_{\mathrm{coc}}
+
\mathcal L_{\mathrm{metric}}
\right)
+
\lambda_{\mathrm{cover}}
\mathcal L_{\mathrm{cover}}.
]

其中：

[
\mathcal L_{\mathrm{geo}}
=========================

\sum_{k,i,j}
q_{ik}q_{jk}
\ell
\left(
d_{g_k}(u_i^{(k)},u_j^{(k)}),
d_G^{\mathrm{loc}}(i,j)
\right).
]

不要再加入：

* 对比负样本；
* 原型对齐；
* 多视图一致性；
* 图重连；
* 超图传播；
* 困难样本挖掘；
* 节点级独立曲率专家。

这些模块会模糊主贡献。

---

# 十一、论文贡献应当这样写

本文贡献不是提出一种新的混合几何模型，而是：

1. **新问题。** 我们首次将图表示中的局部坐标不兼容问题形式化，指出现有多几何方法虽然能够选择局部空间，却不能保证跨区域消息在坐标变换下具有一致含义。

2. **新表示对象。** 我们提出由重叠 chart、局部度量和转换映射构成的可学习图册，使不同局部表示成为同一个图对象的兼容坐标描述。

3. **新传播算子。** 我们提出 Atlas Transport Convolution，在聚合之前通过 transition Jacobian 将跨区域消息运输到兼容坐标中。

4. **新理论。** 我们证明模型对局部坐标重参数化保持不变，并给出由局部失真、转换误差和路径一致性误差共同决定的全局失真上界。

5. **新基准。** 我们构建 Atlas-Het，提供真实 chart、transition 和局部几何标注，并引入坐标干预测试，直接区分真正的图册学习和普通多专家融合。

---

# 十二、48 小时淘汰实验

第一阶段不实现完整模型。

只构造两个或三个重叠 chart，并实现：

1. Euclidean GNN；
2. GraphMoRE 风格几何 MoE；
3. 自由线性 transition 模型；
4. composition-induced transition；
5. Full Atlas Transport。

只观察四项结果：

### 条件一：边界收益

Full Atlas 在三种随机种子下，对边界节点稳定优于 MoE。

建议最低目标：

[
\Delta\operatorname{Acc}_{\mathrm{boundary}}
\geq 2%.
]

### 条件二：transition 必须不可替代

删除 transport 后，边界节点性能和跨 chart distortion 明显恶化。

### 条件三：坐标干预稳定

施加随机坐标变换后：

[
\operatorname{InvErr}*{\mathrm{Atlas}}
\ll
\operatorname{InvErr}*{\mathrm{MoE}}.
]

### 条件四：composition-induced transition 优于自由矩阵

如果自由 (W_{kl}) 与诱导 transition 一样好，说明 atlas 数学结构没有带来实际价值。

任何一项完全不成立，都应立即重新评估，不要继续扩展模块。

---

# 十三、最关键的论文图

论文第一张图不要画三个欧氏、双曲、球面专家。

应该画：

* 左侧：图中存在若干重叠局部区域；
* 中间：每个区域具有自己的坐标网格；
* 跨区域边上的消息不能直接相加；
* transition 将消息运输到同一坐标；
* 右侧：任意改变每个局部坐标后，最终预测不变。

图中直接对比：

### Geometry MoE

独立编码，最后融合。

### GraphAtlas

局部编码，转换运输，坐标无关预测。

一张图就应让审稿人无法将两者混淆。

---

# 十四、最终摘要草案

Graph neural networks commonly assume that node representations share a globally compatible coordinate system. This assumption becomes restrictive on geometrically heterogeneous and heterophilic graphs, where distinct local relational regimes may admit different coordinate descriptions. Existing mixed-geometry methods adaptively select or combine manifold experts, but do not ensure that representations from different local spaces describe the same object under valid coordinate transformations. Consequently, cross-region messages are fused without an explicit notion of coordinate compatibility.

We introduce GraphAtlas, a framework that represents a graph through a collection of overlapping local charts equipped with learnable Riemannian metrics and transition maps. Rather than combining independent geometric experts, GraphAtlas transports messages across chart boundaries using the Jacobians of learned coordinate transformations before neighborhood aggregation. The transition maps are induced by local encoder–decoder pairs and are regularized through cocycle and metric-compatibility constraints. We prove that the resulting atlas transport convolution is invariant to smooth reparameterizations of individual charts. We further derive a global distortion bound determined by local embedding distortion, transition error, and chart-path consistency.

To directly evaluate whether a model recovers compatible local geometries rather than merely increasing model capacity, we introduce Atlas-Het, a controllable benchmark with ground-truth charts, overlaps, coordinate transitions, local metrics, and heterophilic relations. It also contains coordinate-intervention tests that independently reparameterize each local chart while preserving the underlying graph. Experiments on synthetic and real-world heterophilic graphs demonstrate that GraphAtlas improves cross-region representation, boundary-node prediction, and geometric fidelity while remaining stable under arbitrary local coordinate changes.
