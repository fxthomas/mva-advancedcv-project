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
    green|gr-arrow-end|\|\<gtr\>|gr-dash-style-unit|20ln|<graphics||<point|-2|2>|<point|-2|1>|<point|-2|0>|<point|-2|-1>|<point|-2|-2>|<point|-1|-2>|<point|0|-2>|<point|1|-2>|<point|2|-2>|<point|2|-1>|<point|2|0>|<point|2|1>|<point|2|2>|<point|1|2>|<point|0|2>|<point|-1|2>|<point|-1|1>|<point|-1|-1>|<point|1|-1>|<point|1|1>|<with|color|blue|<point|-1|0>>|<with|color|blue|<point|1|0>>|<with|color|blue|<point|0|-1>>|<with|color|blue|<point|0|1>>|<with|color|red|<carc|<point|0.4|-0.4>|<point|-0.4|0.4>|<point|0.3|0.5>>>|<with|color|red|arrow-end|\|\<gtr\>|<line|<point|1|0>|<point|0.2|0.0>>>|<with|color|red|arrow-end|\|\<gtr\>|<line|<point|0|1>|<point|0.0|0.2>>>|<with|color|red|arrow-end|\|\<gtr\>|<line|<point|-1|0>|<point|-0.2|0.0>>>|<with|color|red|arrow-end|\|\<gtr\>|<line|<point|0|-1>|<point|0.0|-0.2>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|2|0>|<point|1.2|0.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|0|2>|<point|0.0|1.3>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|-2|0>|<point|-1.2|0.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|0|-2>|<point|0.0|-1.2>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|1|-1>|<point|0.2|-1.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|-1|-1>|<point|-0.2|-1.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|-1|1>|<point|-0.2|1.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|1|1>|<point|0.2|1.0>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|-1|1>|<point|-1.0|0.2>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|-1|-1>|<point|-1.0|-0.2>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|1|-1>|<point|1.0|-0.2>>>|<with|color|blue|arrow-end|\|\<gtr\>|<line|<point|1|1>|<point|1.0|0.3>>>>>

    <with|font-shape|italic|Image pixels graph showing energy computation for
    the node at the center>
  </center>

  <subsection|Tree-based optimization>

  On the contrary, minimization over a tree can be computed in exactly 2 DP
  passes (forward and backward), and, according to the
  <with|font-shape|italic|SimpleTree> paper, some papers discuss methods
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

  This yields pretty good disparity maps, even on a pair of images that's
  half-blurred and quickly captured by hand with a cellphone (I don't know
  about you, but I consider that to be pretty bad conditions!), such as here
  :

  <center|<image|images/Disparity-FX.png|25%|||>>

  \;
</body>

<\initial>
  <\collection>
    <associate|sfactor|4>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|2|1>>
    <associate|auto-3|<tuple|2.1|2>>
    <associate|auto-4|<tuple|2.2|2>>
    <associate|auto-5|<tuple|2.3|?>>
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
    </associate>
  </collection>
</auxiliary>