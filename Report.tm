<TeXmacs|1.0.7.15>

<style|generic>

<\body>
  <doc-data|<doc-title|Advanced Methods for Computer
  Vision>|<doc-author-data|<author-name|François-Xavier
  Thomas>>|<doc-subtitle|Stereo and image warping>>

  The goal of this project was to make an implementation of and understand
  the theory behind the paper <with|font-shape|italic|A Stereo Approach that
  Handles the Matting Problem via Image Warping>, by Michael Bleyer, Margrit
  Gelautz, Carsten Rother and Christoph Rhemann. As it is quite a long name,
  I'm going to do the same as the authors and refer to it as
  <with|font-shape|italic|WarpMat> in the following document.

  The main part of the paper is the minimization of a complex energy function
  using Loopy Belief Propagation, and having reasonably valid initial guesses
  is primordial for the optimization process.

  The outlined method is therefore the combination of a few other algorithms,
  and depends heavily upon them for the correction of the initial guesses.

  Because of their importance, I'm first going to explain how they work, and
  discuss their performance and implementation, although it's not the main
  point of the paper.

  <section|Image Rectification (<verbatim|src/rectify.py>)>

  Of course, this being a stereo algorithm, it needs rectified images as
  input. I'm not going to expand on this point, as it's been covered
  sufficiently.

  The main reason for this is that, for the following algorithm, you can map
  pixels from the left view to pixels from the right view, by simply
  following a single scanline.

  A simple and naïve automatic SURF-based implementation using OpenCV is
  available in the <verbatim|rectify.py> file, though, just in case.

  \;

  <center|<image|images/Rectify-Scanlines.png|60%|||>>

  <section|Disparity maps (<verbatim|src/simpletree/>)>

  The subject of the disparity maps has also been covered quite extensively,
  but the method <with|font-shape|italic|WarpMat> uses is quite interesting,
  so I'm going to describe it in detail.

  This method is called <with|font-shape|italic|SimpleTree> and comes from
  the paper <with|font-shape|italic|Simple but effective tree structures for
  dynamic programming-based stereo matching>, by the same authors, Michael
  Bleyer and Margrit Gelautz.

  I implemented this algorithm as a Python module, <verbatim|simpletree>. The
  algorithm basically runs the same DP algorithm 8 times on the provided
  images, so the core DP method is written in C, for efficiency, and can be
  found in the <verbatim|simpletree.dp> module.

  <subsection|Energy minimization>

  Disparity map computation is often viewed as a form of energy minimization.
  The energy we seek to minimize here is (<math|D> being the disparity map,
  <math|I<rsub|L>> and <math|I<rsub|R>> the left and right views) :

  <\equation*>
    E<around*|(|D|)>=E<rsub|data><around*|(|D|)> +
    E<rsub|smoothness><around*|(|D|)>=<big|sum><rsub|p\<in\>\<cal-P\>><wide*|m<around*|(|p,d<rsub|p>|)>|\<wide-squnderbrace\>><rsub|<with|mode|text|cost
    of <math|d<rsub|p>>>> + <big|sum><rsub|<around*|(|p,q|)>\<in\>\<cal-N\>><wide*|s<around*|(|d<rsub|p>,d<rsub|q>|)>|\<wide-squnderbrace\>><rsub|<with|mode|text|measures
    how close <math|d<rsub|p>> and <math|d<rsub|q>> are>>
  </equation*>

  <math|d<rsub|p>> is the disparity being assigned to the point <math|p>.
  <math|m<around*|(|p,d<rsub|p>|)>> is a measure of how bad this assignment
  is.

  Finally, <math|s<around*|(|d<rsub|p>,d<rsub|q>|)>> is a function of the
  difference between <math|d<rsub|p>> and <math|d<rsub|q>>, as well as
  <math|I<rsub|L><around*|(|p|)>> and <math|I<rsub|R><around*|(|p|)>>.

  <subsection|Message passing>

  Using a message-passing algorithm, this minimization can be computed as the
  optimization through a connected graph composed of all the pixels of the
  image, but this has a very high complexity, as it involves looping through
  the (cyclic) graph multiple times, and is not even an exact minimization
  because of it.

  <\center>
    <with|gr-mode|<tuple|edit|spline>|gr-frame|<tuple|scale|1cm|<tuple|0.5gw|0.5gh>>|gr-geometry|<tuple|geometry|0.393357par|0.366676par|center>|gr-grid|<tuple|empty>|gr-grid-old|<tuple|cartesian|<point|0|0>|1>|gr-edit-grid-aspect|<tuple|<tuple|axes|none>|<tuple|1|none>|<tuple|10|none>>|gr-edit-grid|<tuple|empty>|gr-edit-grid-old|<tuple|cartesian|<point|0|0>|1>|gr-color|dark
    blue|gr-arrow-end|\|\<gtr\>|gr-dash-style-unit|20ln|<graphics||<point|-2|2>|<point|-2|1>|<point|-2|0>|<point|-2|-1>|<point|-2|-2>|<point|-1|-2>|<point|0|-2>|<point|1|-2>|<point|2|-2>|<point|2|-1>|<point|2|0>|<point|2|1>|<point|2|2>|<point|1|2>|<point|0|2>|<point|-1|2>|<point|-1|1>|<point|-1|-1>|<point|1|-1>|<point|1|1>|<with|color|blue|<point|-1|0>>|<with|color|blue|<point|1|0>>|<with|color|blue|<point|0|-1>>|<with|color|blue|<point|0|1>>|<with|color|red|<carc|<point|0.4|-0.4>|<point|-0.4|0.4>|<point|0.3|0.5>>>|<with|color|red|arrow-end|\|\<gtr\>|<line|<point|1|0>|<point|0.2|0.0>>>|<with|color|red|arrow-end|\|\<gtr\>|<line|<point|0|1>|<point|0.0|0.2>>>|<with|color|red|arrow-end|\|\<gtr\>|<line|<point|-1|0>|<point|-0.2|0.0>>>|<with|color|red|arrow-end|\|\<gtr\>|<line|<point|0|-1>|<point|0.0|-0.2>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|2|0>|<point|1.2|0.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|0|2>|<point|0.0|1.3>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|-2|0>|<point|-1.2|0.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|0|-2>|<point|0.0|-1.2>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|1|-1>|<point|0.2|-1.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|-1|-1>|<point|-0.2|-1.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|-1|1>|<point|-0.2|1.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|1|1>|<point|0.2|1.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|-1|1>|<point|-1.0|0.2>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|-1|-1>|<point|-1.0|-0.2>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|1|-1>|<point|1.0|-0.2>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|1|1>|<point|1.0|0.3>>>|<with|color|dark
    blue|arrow-end|\|\<gtr\>|dash-style-unit|20ln|<spline|<point|0|0.2>|<point|0.369759921654311|0.451450849617079>|<point|0.0|1.0>>>|<with|color|dark
    blue|arrow-end|\|\<gtr\>|dash-style-unit|20ln|<spline|<point|0|-0.2>|<point|0.306397893974115|-0.471169854257612>|<point|0.0|-1.0>>>|<with|color|dark
    blue|arrow-end|\|\<gtr\>|dash-style-unit|20ln|<spline|<point|0.2|0>|<point|0.502207008235756|0.296224419260636>|<point|1.0|0.0>>>|<with|color|dark
    blue|arrow-end|\|\<gtr\>|dash-style-unit|20ln|<spline|<point|-0.2|0>|<point|-0.477689265243385|-0.270390410300793>|<point|-1.0|0.0>>>>>

    <with|font-shape|italic|Image pixels graph showing energy computation for
    the node at the center>
  </center>

  <subsection|Tree-based optimization>

  On the contrary, minimization over a tree can be computed in exactly 2 DP
  passes (forward and backward), and, according to the
  <with|font-shape|italic|SimpleTree> paper, some authors discussed methods
  using different kinds of trees.

  <with|font-shape|italic|SimpleTree>, similarly, uses a tree that can be
  represented as :

  <center|<with|gr-mode|<tuple|group-edit|move>|gr-frame|<tuple|scale|1cm|<tuple|0.5gw|0.480002gh>>|gr-geometry|<tuple|geometry|0.373358par|0.400008par|center>|gr-grid|<tuple|empty>|gr-grid-old|<tuple|cartesian|<point|0|0>|1>|gr-edit-grid-aspect|<tuple|<tuple|axes|none>|<tuple|1|none>|<tuple|10|none>>|gr-edit-grid|<tuple|empty>|gr-edit-grid-old|<tuple|cartesian|<point|0|0>|1>|gr-color|orange|gr-arrow-end|\|\<gtr\>|gr-line-width|5ln|<graphics||<point|-2|2>|<point|-1|2>|<point|0|2>|<point|1|2>|<point|2|2>|<point|2|1>|<point|1|1>|<point|0|1>|<point|-1|1>|<point|-2|1>|<point|-2|0>|<point|-1|0>|<point|0|0>|<point|1|0>|<point|2|0>|<point|2|-1>|<point|1|-1>|<point|0|-1>|<point|-1|-1>|<point|-2|-1>|<point|-2|-2>|<point|-1|-2>|<point|0|-2>|<point|1|-2>|<point|2|-2>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|-2|-2>|<point|-0.2|-2.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|2|-2>|<point|0.2|-2.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|2|-1>|<point|0.2|-1.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|2|0>|<point|0.2|0.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|2|1>|<point|0.2|1.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|2|2>|<point|0.2|2.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|-2|2>|<point|-0.2|2.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|-2|1>|<point|-0.2|1.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|-2|0>|<point|-0.2|0.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|-2|-1>|<point|-0.2|-1.0>>>|<with|color|orange|arrow-end|\|\<gtr\>|line-width|5ln|<line|<point|0|2>|<point|0.0|0.2>>>|<with|color|orange|arrow-end|\|\<gtr\>|line-width|5ln|<line|<point|0|-2>|<point|0.0|-0.2>>>|<with|color|blue|<text-at|Horizontal
  scanlines|<point|-1.6|-2.6>>>|<with|color|orange|<text-at|Vertical message
  passing|<point|-1.9|2.6>>>>>>

  As you can see, the disparity for the horizontal scanlines are
  <with|font-shape|italic|first> optimized
  <with|font-shape|italic|independently> from each other in one set of DP
  passes (forward-backward), which, as earlier works proved, yields very
  noisy results (fig. a).

  <\center>
    <image|images/Disparity-THorzScanlines.png|25%|||><image|images/Disparity-THorzOnly.png|25%|||>

    <with|font-shape|italic|(a) Horizontal scanlines only <emdash> vs
    <emdash> (b) Whole tree optimization>
  </center>

  Then, as we have the minimizing disparity for each scanline, we optimize
  these disparity vertically, effectively optimizing the whole tree (fig. b).

  Only using one vertical link between each scanline has drawbacks, though :
  as you can see in b., there are still some artifacts, as the disparities
  are sometimes not vertically continuous.

  To prevent that, <with|font-shape|italic|SimpleTree> uses the transposed,
  vertical version of the same tree to pre-compute messages
  <math|m<rprime|'>> that act as input for the above algorithm instead of
  <math|m>.

  <\center>
    <with|gr-mode|<tuple|group-edit|zoom>|gr-frame|<tuple|scale|1cm|<tuple|0.5gw|0.480002gh>>|gr-geometry|<tuple|geometry|0.193365par|0.180017par|center>|gr-grid|<tuple|empty>|gr-grid-old|<tuple|cartesian|<point|0|0>|1>|gr-edit-grid-aspect|<tuple|<tuple|axes|none>|<tuple|1|none>|<tuple|10|none>>|gr-edit-grid|<tuple|empty>|gr-edit-grid-old|<tuple|cartesian|<point|0|0>|1>|gr-color|orange|gr-arrow-end|\|\<gtr\>|gr-line-width|5ln|<graphics||<with|magnify|0.486543580360217|<point|0.975460653997152|0.481767434616943>>|<with|magnify|0.486543580360217|<point|0.970690281469981|-0.491308033116627>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|0.481767361339617|-0.975460580719829>|<point|0.486060696614068|-0.0996926597596162>>>|<with|magnify|0.486543580360217|<point|0.486537733866784|-0.00238511298625913>>|<with|magnify|0.486543580360217|<point|0.9683050952064|-0.977845766983411>>|<with|magnify|0.486543580360217|<point|-0.00238518626358454|-0.486537660589458>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|0.00477037252716918|0.973075541010894>|<point|4.77037252716951e-4|0.0973076200506827>>>|<with|magnify|0.486543580360217|<point|-0.486537733866784|0.00238525954091004>>|<with|magnify|0.486543580360217|<point|0.491308106393952|0.970690354747304>>|<with|magnify|0.486543580360217|<point|0.00477037252716918|0.973075541010894>>|<with|magnify|0.486543580360217|<point|0.973075467733571|-0.0047702992498437>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|-0.491308106393952|-0.970690208192658>|<point|-0.487014771119499|-0.0949222872324477>>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|0.9683050952064|-0.977845766983411>|<point|0.972598430480853|-0.1020778460232>>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|0.977845840260733|0.968305168483723>|<point|0.97355250498628|0.0925372475235133>>>|<with|magnify|0.486543580360217|<point|0.00238518626358462|0.486537807144109>>|<with|magnify|0.486543580360217|<point|-0.488922920130367|-0.484152474325874>>|<with|magnify|0.486543580360217|<point|-0.975460653997152|-0.481767288062291>>|<with|magnify|0.486543580360217|<point|-0.484152547603199|0.488922993407692>>|<with|magnify|0.486543580360217|<point|-0.491308106393952|-0.970690208192658>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|-0.00477037252716912|-0.973075394456248>|<point|-4.7703725271688e-4|-0.0973074734960315>>>|<with|magnify|0.486543580360217|<point|0.484152547603199|-0.488922846853041>>|<with|magnify|0.486543580360217|<point|-0.481767361339617|0.975460727274475>>|<with|magnify|0.486543580360217|<point|0.977845840260733|0.968305168483723>>|<with|magnify|0.486543580360217|<point|-0.973075467733571|0.0047704458044946>>|<with|magnify|0.486543580360217|<point|0.488922920130368|0.484152620880525>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|-0.9683050952064|0.977845913538057>|<point|-0.972598430480853|0.102077992577851>>>|<with|magnify|0.486543580360217|<point|-0.977845840260733|-0.968305021929077>>|<with|magnify|0.486543580360217|<point|-0.9683050952064|0.977845913538057>>|<with|magnify|0.486543580360217|color|orange|arrow-end|\|\<gtr\>|line-width|5ln|<line|<point|0.973075467733571|-0.0047702992498437>|<point|0.0973075467733571|-4.76963975391461e-4>>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|0.491308106393952|0.970690354747304>|<point|0.487014771119499|0.094922433787098>>>|<with|magnify|0.486543580360217|<point|-0.00477037252716912|-0.973075394456248>>|<with|magnify|0.486543580360217|<point|-0.970690281469981|0.491308179671278>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|-0.481767361339617|0.975460727274475>|<point|-0.486060696614068|0.0996928063142674>>>|<with|magnify|0.486543580360217|<point|0.481767361339617|-0.975460580719829>>|<with|magnify|0.486543580360217|color|orange|arrow-end|\|\<gtr\>|line-width|5ln|<line|<point|-0.973075467733571|0.0047704458044946>|<point|-0.0973075467733571|4.77110530042371e-4>>>|<with|magnify|0.486543580360217|<point|3.48368831167637e-17|3.7677263818031e-5>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|-0.977845840260733|-0.968305021929077>|<point|-0.97355250498628|-0.0925371009688622>>>>><math|\<longrightarrow\>>
    <with|gr-mode|<tuple|group-edit|rotate>|gr-frame|<tuple|scale|1cm|<tuple|0.5gw|0.480002gh>>|gr-geometry|<tuple|geometry|0.193365par|0.180017par|center>|gr-grid|<tuple|empty>|gr-grid-old|<tuple|cartesian|<point|0|0>|1>|gr-edit-grid-aspect|<tuple|<tuple|axes|none>|<tuple|1|none>|<tuple|10|none>>|gr-edit-grid|<tuple|empty>|gr-edit-grid-old|<tuple|cartesian|<point|0|0>|1>|gr-color|orange|gr-arrow-end|\|\<gtr\>|gr-line-width|5ln|<graphics||<with|magnify|0.486543580360217|<point|-0.485172695128081|0.973771858146385>>|<with|magnify|0.486543580360217|color|orange|arrow-end|\|\<gtr\>|line-width|5ln|<line|<point|-0.00136886843854398|-0.973085353440051>|<point|-1.36196162606936e-4|-0.0973077762951367>>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|0.973086964251292|-0.00136879247452092>|<point|0.0973093871063764|-1.36120198583877e-4>>>|<with|magnify|0.486543580360217|<point|-0.487227148921305|-0.485857437095131>>|<with|magnify|0.486543580360217|<point|-3.68363332425945e-5|7.11969088412984e-7>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|-0.972400611472989|0.487913577663621>|<point|-0.0966230343280837|0.486680905387683>>>|<with|magnify|0.486543580360217|<point|-0.973085429404071|0.00137047924978361>>|<with|magnify|0.486543580360217|<point|-0.487911966852382|-0.972400535508968>>|<with|magnify|0.486543580360217|<point|0.486543865837447|-6.83974543444776e-4>>|<with|magnify|0.486543580360217|<point|0.485859047906371|-0.487227072957283>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|-0.973085429404071|0.00137047924978361>|<point|-0.09730785225916|1.37806973846576e-4>>>|<with|magnify|0.486543580360217|<point|0.973086964251292|-0.00136879247452092>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|-0.974455065266217|-0.971715717577896>|<point|-0.0986774881213114|-0.972948389853834>>>|<with|magnify|0.486543580360217|<point|-0.971715793541917|0.974456676077457>>|<with|magnify|0.486543580360217|<point|-0.973770247335143|-0.485172619164058>>|<with|magnify|0.486543580360217|<point|-0.00136886843854398|-0.973085353440051>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|0.972402146320211|-0.487911890888359>|<point|0.0966245691753009|-0.486679218612422>>>|<with|magnify|0.486543580360217|<point|0.487913501699599|0.972402222284231>>|<with|magnify|0.486543580360217|<point|0.971717328389137|-0.974454989302196>>|<with|magnify|0.486543580360217|<point|0.974456600113438|0.971717404353159>>|<with|magnify|0.486543580360217|<point|-0.972400611472989|0.487913577663621>>|<with|magnify|0.486543580360217|<point|0.00137040328576056|0.973087040215312>>|<with|magnify|0.486543580360217|<point|0.485174229975297|-0.973770171371122>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|0.974456600113438|0.971717404353159>|<point|0.0986790229685275|0.972950076629096>>>|<with|magnify|0.486543580360217|<point|-6.84050507467853e-4|-0.486542255026208>>|<with|magnify|0.486543580360217|<point|0.973771782182365|0.485174305939321>>|<with|magnify|0.486543580360217|<point|0.487228683768522|0.485859123870394>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|0.971717328389137|-0.974454989302196>|<point|0.0959397512442238|-0.973222317026258>>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|0.973771782182365|0.485174305939321>|<point|0.0979942050374531|0.486406978215255>>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|-0.971715793541917|0.974456676077457>|<point|-0.0959382163970075|0.97322400380152>>>|<with|magnify|0.486543580360217|<point|-0.974455065266217|-0.971715717577896>>|<with|magnify|0.486543580360217|<point|-0.48654233099023|6.85661318707495e-4>>|<with|magnify|0.486543580360217|<point|-0.485857513059156|0.487228759732546>>|<with|magnify|0.486543580360217|color|blue|arrow-end|\|\<gtr\>|<line|<point|-0.973770247335143|-0.485172619164058>|<point|-0.097992670190236|-0.486405291439994>>>|<with|magnify|0.486543580360217|color|orange|arrow-end|\|\<gtr\>|line-width|5ln|<line|<point|0.00137040328576056|0.973087040215312>|<point|1.37731009823518e-4|0.0973094630703995>>>|<with|magnify|0.486543580360217|<point|0.972402146320211|-0.487911890888359>>|<with|magnify|0.486543580360217|<point|6.85585354684436e-4|0.48654394180147>>>>

    <with|font-shape|italic|Vertical Tree <math|\<longrightarrow\>>
    Horizontal Tree>
  </center>

  <\equation*>
    m<rprime|'><around*|(|p,d|)>=m<around*|(|p,d|)>+\<lambda\>*<around*|(|V<around*|(|p,d|)>-min<rsub|i\<in\>\<cal-D\>>V<around*|(|p,i|)>|)>
  </equation*>

  Concerning the implementation, all these DP passes -- 4 times
  forward-backward, so 8 passes -- are in fact the same algorithm, and can be
  factored. My implementation is a C module (for efficiency) that's called 8
  times with different energy values and scanline axes.

  The algorithm itself is quite fast, my unoptimized implementation taking
  2.1 seconds for a Tsukuba image pair on a single thread, and 1.5 seconds
  with OpenMP on 4 threads -- CPU is a <with|font-shape|italic|Core i5>
  1.7GHz --, but this could certainly be improved.

  This also yields pretty good disparity maps in a few seconds for
  <math|600\<times\>400> images, even on a pair of images that's half-blurred
  and quickly captured by hand with a cellphone camera (I don't know about
  you, but I consider that to be pretty bad conditions!), such as here :

  <center|<image|images/Disparity-FX.png|25%|||>>

  <section|Segmentation (<verbatim|src/meanshift/>)>

  Another input the <with|font-shape|italic|WarpMat> algorithm needs is an
  over-segmented left view. There are a lot of algorithms available that do
  it, but <with|font-shape|italic|WarpMat> recommends the Mean-Shift
  algorithm.

  The basic premise behind Mean-Shift is the following : each pixel <math|p>
  is surrounded by others in a known radius, and the mean of all these
  surrounding pixels is <math|m>. Each iteration of Mean-Shift then moves
  <math|p> towards <math|m>. After a few iterations, all the pixels have
  converged towards the center of the local cluster where they belong. The
  image is then segmented in regions of pixels with the same center.

  <center|<with|gr-mode|<tuple|edit|point>|gr-frame|<tuple|scale|1cm|<tuple|0.549995gw|0.330016gh>>|gr-geometry|<tuple|geometry|0.606676par|0.233344par|center>|gr-line-width|2ln|gr-arrow-end|\|\<gtr\>|gr-point-style|square|<graphics||<with|color|green|<point|-2.6738|1.10706>>|<with|color|green|<point|-2.63993|0.53132>>|<with|color|green|<point|-1.96259|0.785322>>|<with|color|green|<point|-2.18273|-0.40002>>|<with|color|green|<point|-2.877|-0.0444173>>|<with|color|green|<point|-1.48846|0.0571835>>|<with|color|green|<point|-1.09899|0.751455>>|<with|color|dark
  green|<point|-2.40286|0.04025>>|<with|color|dark
  green|<point|-1.94566|0.226518>>|<with|color|dark
  green|<point|-3.0294|0.599054>>|<with|color|dark
  green|<point|-2.19966|1.19173>>|<with|color|dark
  green|<point|-1.33606|1.22559>>|<with|color|dark
  green|<point|-1.50539|-0.40002>>|<with|color|yellow|<point|-1.08205|0.0910504>>|<with|color|yellow|<point|-1.04819|-0.535487>>|<with|color|yellow|<point|-0.624851|0.141851>>|<with|color|yellow|<point|-0.709518|1.32719>>|<with|color|yellow|<point|-1.03125|1.7336>>|<with|color|yellow|<point|-1.62393|1.59813>>|<with|color|yellow|<point|-1.62393|0.836123>>|<with|color|orange|<point|-0.438583|0.565187>>|<with|color|orange|<point|-0.658718|0.802256>>|<with|color|orange|<point|-0.303115|1.37799>>|<with|color|orange|<point|-0.47245|1.7336>>|<with|color|orange|<point|-0.387783|-0.416953>>|<with|color|orange|<point|-0.099914|0.107984>>|<with|color|red|<point|0.0524871|1.02239>>|<with|color|red|<point|0.323422|0.615988>>|<with|color|red|<point|0.306489|-0.0613507>>|<with|color|red|<point|-0.0660471|-0.535487>>|<with|color|red|<point|1.3225|0.666788>>|<with|color|red|<point|0.763692|1.24253>>|<with|color|red|<point|0.171021|1.37799>>|<with|color|red|<point|0.374223|1.86906>>|<with|color|red|<point|0.831426|0.514387>>|<with|color|red|<point|0.933027|-0.416953>>|<with|color|red|<point|0.50969|-0.349219>>|<with|color|red|<point|1.4241|-0.0952176>>|<with|color|dark
  magenta|<point|0.374223|1.03932>>|<with|color|dark
  magenta|<point|0.712892|0.107984>>|<with|color|dark
  magenta|<point|1.06849|0.107984>>|<with|color|dark
  magenta|<point|1.06849|0.836123>>|<with|color|dark
  magenta|<point|0.797559|1.76746>>|<with|color|green|line-width|2ln|<carc|<point|-2.877|-0.0444173>|<point|-1.09899|0.751455>|<point|-1.08205|0.0910504>>>|<with|color|red|line-width|2ln|<carc|<point|0.933027|-0.416953>|<point|0.374223|1.86906>|<point|-0.48938351633814|0.836122502976584>>>|<with|color|red|<text-at|Red
  cluster|<point|1.91517|-0.40002>>>|<with|color|green|<text-at|Green
  cluster|<point|-4.65501|-0.755622>>>|<with|arrow-end|\|\<gtr\>|line-width|2ln|<line|<point|-1.82711|-0.5606>|<point|-2.04726154253208|0.429719539621643>>>|<with|arrow-end|\|\<gtr\>|line-width|2ln|<line|<point|-1.18302|-0.122661>|<point|-1.82712660404815|0.463586453234555>>>|<with|arrow-end|\|\<gtr\>|line-width|2ln|<line|<point|-2.44827|1.26778>|<point|-2.08112845614499|0.632921021299114>>>|<with|arrow-end|\|\<gtr\>|line-width|2ln|<line|<point|-2.95483|0.165084>|<point|-2.25046302420955|0.463586453234555>>>|<with|arrow-end|\|\<gtr\>|line-width|2ln|<line|<point|-1.09899|0.751455>|<point|-1.72552586320942|0.615987564492658>>>|<with|arrow-end|\|\<gtr\>|line-width|2ln|<line|<point|1.80652|0.381793>|<point|0.746758830533139|0.802255589363672>>>|<with|arrow-end|\|\<gtr\>|line-width|2ln|<line|<point|1.33134|1.71579>|<point|0.848359571371875|1.07319089826697>>>|<with|arrow-end|\|\<gtr\>|line-width|2ln|<line|<point|-0.338874|0.149422>|<point|0.526623892049213|0.700654848524937>>>|<with|arrow-end|\|\<gtr\>|line-width|2ln|<line|<point|0.698132|-0.443683>|<point|0.628224632887948|0.514386823653922>>>|<with|arrow-end|\|\<gtr\>|line-width|2ln|<line|<point|0.0552087|1.72883>|<point|0.57742426246858|0.954656700621775>>>>>>

  The corresponding Python module, <verbatim|meanshift>, is a light wrapper
  around a small part of the source of the EDISON system implemented by D.
  Comanicu and P. Meer, which can be found at their website. This system
  integrates a few improvements over the original mean-shift algorithm,
  described in detail in <with|font-shape|italic|Mean Shift: A Robust
  Approach Toward Feature Space Analysis> by the same people.

  <center|<image|images/Segmentation-Labels.png|40%|||>>

  The segmentation module computes a segmented Tsukuba view in approx. 1.6
  seconds on my machine.

  <section|Plane fitting (<verbatim|src/warpmat.py>)>

  For <with|font-shape|italic|WarpMat>, the disparity planes are modelled is
  a piecewise-affine manner for each segment of the segmented image, such as
  in this simple example (fitted planes are represented as the blue, green
  and red lines) :

  <center|<image|images/Disparity-PlaneModels.png|35%|||>>

  The initialization is done by linear regression on all the pixels of each
  segment, and the naive Python implementation takes about 3.9 seconds for
  the Tsukuba pair, for approx. 8k labels.

  <section|Artificial right view generation (<verbatim|src/warpmat.py>)>

  Given the disparity map, and the segments in the left image, we can
  reconstruct the right image by simply translating iso-disparity components.

  The greater the disparity, the greater the displacement, as in the example
  below. The artificial right view is then computed by blending the pixel
  values together.

  \;

  <center|<with|gr-mode|<tuple|edit|line>|gr-frame|<tuple|scale|0.75cm|<tuple|0.0900389gw|0.0950185gh>>|gr-geometry|<tuple|geometry|0.480003par|0.280004par|center>|gr-grid|<tuple|empty>|gr-grid-old|<tuple|cartesian|<point|0|0>|1>|gr-edit-grid-aspect|<tuple|<tuple|axes|none>|<tuple|1|none>|<tuple|10|none>>|gr-edit-grid|<tuple|empty>|gr-edit-grid-old|<tuple|cartesian|<point|0|0>|1>|magnify|0.75|gr-arrow-end|\|\<gtr\>|<graphics||<text-at|<math|d>|<point|-0.4|4.3>>|<text-at|Image
  pixels (Left view)|<point|4.71756184680513|-0.387180844026988>>|<with|color|red|line-width|10ln|<line|<point|1|1>|<point|3.0|1.0>>>|<with|color|blue|line-width|10ln|<line|<point|3|2>|<point|4.0|2.0>>>|<with|color|orange|line-width|10ln|<line|<point|4|4>|<point|5.4|4.0>>>|<with|color|dark
  green|line-width|10ln|<line|<point|6|1>|<point|7.0|1.0>>>|<with|color|dark
  green|arrow-end|\|\<gtr\>|<line|<point|7|1>|<point|7.5|1.0>>>|<with|color|red|arrow-end|\|\<gtr\>|<line|<point|3|1>|<point|3.6|1.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|4|2>|<point|5.5|2.0>>>|<with|color|orange|arrow-end|\|\<gtr\>|<line|<point|5.4|4>|<point|8.4|4.0>>>|<with|arrow-end|\|\<gtr\>|<line|<point|0|0>|<point|0.0|5.0>>>|<with|arrow-end|\|\<gtr\>|<line|<point|0|0>|<point|8.58424835736649|-0.00509326630506681>>>>><center|<with|gr-mode|<tuple|edit|line>|gr-frame|<tuple|scale|0.75cm|<tuple|0.0700408gw|0.0950185gh>>|gr-geometry|<tuple|geometry|0.480003par|0.280004par|center>|gr-grid|<tuple|empty>|gr-grid-old|<tuple|cartesian|<point|0|0>|1>|gr-edit-grid-aspect|<tuple|<tuple|axes|none>|<tuple|1|none>|<tuple|10|none>>|gr-edit-grid|<tuple|empty>|gr-edit-grid-old|<tuple|cartesian|<point|0|0>|1>|magnify|0.75|gr-arrow-end|\|\<gtr\>|<graphics||<text-at|<math|d>|<point|-0.4|4.3>>|<text-at|Image
  pixels (Reconstructed right view)|<point|2.50653746086343|-0.361154473695815>>|<with|color|red|line-width|10ln|<line|<point|1.6|1.0>|<point|3.6|1.0>>>|<with|color|blue|line-width|10ln|<line|<point|4.5|2.0>|<point|5.5|2.0>>>|<with|color|orange|line-width|10ln|<line|<point|6.9|4.1>|<point|8.3|4.1>>>|<with|color|dark
  green|line-width|10ln|<line|<point|6.6|1.0>|<point|7.6|1.0>>>|<with|color|orange|line-width|10ln|<line|<point|6.9|4.1>|<point|8.3|4.1>>>|<with|arrow-end|\|\<gtr\>|<line|<point|0|0>|<point|0.0|5.0>>>|<with|arrow-end|\|\<gtr\>|<line|<point|0|0>|<point|8.8066763681263|0.0209463332892358>>>>>>>

  <image|images/Disparity-RightReconstruction.png|50%|||>

  The left image can be reconstructed with the same idea, without displacing
  the disparity components.

  <section|<with|font-shape|italic|WarpMat>>

  Tying all these different parts together, we can now build the
  <with|font-shape|italic|WarpMat> algorithm. We have our initial
  <math|D<rsub|S>> (the disparity models for each segment <math|S>) and
  <math|c<rsub|p>,\<alpha\><rsub|p><rsub|>> (color and opacity) for each
  point <math|p> in the image.

  What WarpMat does, roughly, is enumerating a set of
  <math|D<rsub|S>,c<rsub|p>,\<alpha\><rsub|p>> values for each point and
  segment, and finding the values minimizing a global energy :

  <\enumerate>
    <item>Find <math|D<rsub|S>> by minimizing the following energy :

    <\equation*>
      E<rsub|total> = E<rsub|l>+E<rsub|asum>+E<rsub|r>+E<rsub|asmooth>+E<rsub|dsmooth>
    </equation*>

    <\itemize>
      <item><math|E<rsub|l>> is minimized when the reconstructed left image
      is the same as the real left view

      <item><math|E<rsub|r>> is minimized when the reconstructed right image
      is the same as the real right view

      <item><math|E<rsub|asum>> is infinite if, for at least one point
      <math|p> in the reconstructed right image, the sum of
      <math|\<alpha\><rsub|p,d>> values over each disparity label is not
      equal to 1. This ensures that there is no overflow nor underflow in
      pixel values.

      <item><math|E<rsub|asmooth>> and <math|E<rsub|dsmooth>> respectively
      minimize the <math|\<alpha\>> and disparity plane differences between
      segments
    </itemize>

    <with|font-shape|italic|(The formulas for these energies can be found in
    the paper, I chose not to include them here, as they don't help very much
    the basic understanding of the algorithm)>

    \;

    The energy being pretty complicated, <with|font-shape|italic|WarpMat>
    finds a set of candidate disparity planes and iterates through it to find
    the best value.

    <item>For each point <math|p>, find <math|c<rsub|p>> and
    <math|\<alpha\><rsub|p>> (color and opacity) minimizing
    <math|E<rsub|asum>,E<rsub|l>> and <math|E<rsub|r>>, by using belief
    propagation.

    <item>Repeat until satisfied
  </enumerate>

  <center|<with|gr-mode|<tuple|group-edit|move>|gr-frame|<tuple|scale|1cm|<tuple|0.5gw|-5442tmpt>>|gr-geometry|<tuple|geometry|0.666669par|0.266669par|center>|gr-grid|<tuple|empty>|gr-grid-old|<tuple|cartesian|<point|0|0>|1>|gr-edit-grid-aspect|<tuple|<tuple|axes|none>|<tuple|1|none>|<tuple|10|none>>|gr-edit-grid|<tuple|empty>|gr-edit-grid-old|<tuple|cartesian|<point|0|0>|1>|gr-arrow-end|\|\<gtr\>|gr-color|blue|<graphics||<cline|<point|-4|2.5>|<point|-1.0|2.5>|<point|-1.0|1.6>|<point|-4.0|1.6>>|<with|magnify|1.35726900069715|<cline|<point|0.464096498954279|2.66077105031372>|<point|0.464096498954279|1.43922894968628>|<point|4.53590350104572|1.43922894968628>|<point|4.53590350104572|2.66077105031372>>>|<text-at|Disparity
  model|<point|-3.77260732107421|1.92719748908586>>|<text-at|Segment colors
  and <math|\<alpha\>>|<point|0.79030064317682|1.92793127000926>>|<with|color|blue|arrow-end|\|\<gtr\>|<spline|<point|-2.39646|2.5>|<point|-0.0390097896547162|3.18115822198704>|<point|2.58280460878253|2.66077105031372>>>|<with|color|blue|arrow-end|\|\<gtr\>|<spline|<point|2.52971|1.43923>|<point|-0.250677999735415|0.937475195131631>|<point|-2.55627078444612|1.6>>>|<with|color|blue|<text-at|optimization
  with constant <math|D<rsub|S>>|<point|-2.24035884058738|3.51982807381929>>>|<with|color|blue|<text-at|optimization
  with constant <math|c<rsub|p>> and <math|\<alpha\><rsub|p>>|<point|-2.62136332715968|0.429471225029766>>>>>>

  \;

  Overall, the core of the <with|font-shape|italic|WarpMat> optimization is
  akin to a brute-force method with some hacks to make it run faster, and
  doesn't really introduce new concepts and ideas (at least in my opinion --
  feel free to disagree). And (according to the authors) it doesn't even run
  very fast, as they claim it processed a Tsukuba pair in approx. 10 minutes.

  The ideas and methods surrounding the algorithm are what's interesting
  here. In summary, here are the main interesting points :

  <\itemize>
    <item>DP-based disparity computation

    <item>Mean-Shift segmentation

    <item>Optimization of a piecewise-affine disparity model

    <item>Possibility of reconstructing an artificial right view, based on
    the disparity
  </itemize>
</body>

<\initial>
  <\collection>
    <associate|sfactor|4>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-10|<tuple|6|6>>
    <associate|auto-2|<tuple|2|2>>
    <associate|auto-3|<tuple|2.1|2>>
    <associate|auto-4|<tuple|2.2|2>>
    <associate|auto-5|<tuple|2.3|3>>
    <associate|auto-6|<tuple|3|4>>
    <associate|auto-7|<tuple|4|4>>
    <associate|auto-8|<tuple|5|5>>
    <associate|auto-9|<tuple|6|6>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Image
      Rectification (<with|font-family|<quote|tt>|language|<quote|verbatim>|src/rectify.py>)>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Disparity
      maps (<with|font-family|<quote|tt>|language|<quote|verbatim>|src/simpletree/>)>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <with|par-left|<quote|1.5fn>|Energy minimization
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1.5fn>|Message passing
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1.5fn>|Tree-based optimization
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|1.5fn>| <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Segmentation
      (<with|font-family|<quote|tt>|language|<quote|verbatim>|src/meanshift/>)>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Plane
      fitting (<with|font-family|<quote|tt>|language|<quote|verbatim>|src/warpmat.py>)>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Artificial
      right view generation (<with|font-family|<quote|tt>|language|<quote|verbatim>|src/warpmat.py>)>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|<with|font-shape|<quote|italic>|WarpMat>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-10><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>