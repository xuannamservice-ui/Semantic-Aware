import re, os

tex_path = "access.tex"
with open(tex_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update text passages where citations should be streamlined
# Table 1 novelty
text = text.replace(
    r"\cite{noll1976zernike, roddier1990zernike, andrews2005laser, tse2005fundamentals}",
    r"\cite{noll1976zernike, andrews2005laser}"
)
text = text.replace(
    r"\cite{hedayat2007linear, mehana2012mmse, huang2018spatialmode, li2022sic, amhoud2019modeselection, ahmadihesar2025joint}",
    r"\cite{hedayat2007linear, ahmadihesar2025joint}"
)
text = text.replace(
    r"\cite{ma2024bitnet158, bengio2013ste, hinton2015distillation, kplanes2023}",
    r"\cite{ma2024bitnet158, bengio2013ste, kplanes2023}"
)
text = text.replace(
    r"\cite{liang2025semimportance, abdelkhalek2012uep}",
    r"\cite{liang2025semimportance}"
)

# Section I
text = text.replace(
    r"causing severe intermodal crosstalk~\cite{anguita2008crosstalk, amhoud2020unified}.",
    r"causing severe intermodal crosstalk~\cite{lin2025oamsk}."
)

# Section II
old_sec2_p1 = r"""Volumetric video transmission has traditionally centered on point clouds and dynamic meshes~\cite{enenche2025metaverse}. While DeepJSCC has recently been explored for NeRF image feature streaming~\cite{cao2025citynerf}, 3DGS~\cite{kerbl20233dgs} enables direct rasterization without volumetric ray-marching, and other works addressed point cloud concealment~\cite{hung2022pcconceal}. Contemporary advances in semantic 3DGS transmission include HiGS-3DSC~\cite{higs2026}, 3D Gaussian SLAM~\cite{gaussianslam2026}, and RIS-3DGS~\cite{ris3dgs2025}, while packet-loss resilience was investigated in~\cite{tao2026packetloss}. DeepJSCC~\cite{bourtsoulatze2019deepjscc} and generative semantic communications~\cite{zhang2025semcomedge, ren2026gensemcom, chen2025gansemantic, tung2025trangdeep} reconstruct corrupted content gracefully. Importance-aware UEP allocates protection across spatial modes~\cite{liang2025semimportance} or wireless/FSO links~\cite{abdelkhalek2012uep, huo2016fsouep}, but requires instantaneous transmit CSI. In contrast, our scheme replicates Semantic Anchors over link-statistics-derived mode pairs without instantaneous CSI feedback."""

new_sec2_p1 = r"""Volumetric video transmission has evolved from point clouds toward 3DGS~\cite{kerbl20233dgs}, enabling direct rasterization without volumetric ray-marching. Contemporary advances in semantic 3DGS transmission include HiGS-3DSC~\cite{higs2026}, 3D Gaussian SLAM~\cite{gaussianslam2026}, and RIS-assisted delivery~\cite{ris3dgs2025}, while packet-loss resilience and adaptive streaming were investigated in~\cite{tao2026packetloss, wan2026aacgs}. DeepJSCC~\cite{bourtsoulatze2019deepjscc} and generative semantic communications~\cite{zhang2025semcomedge, ren2026gensemcom, tung2025trangdeep} reconstruct corrupted content gracefully. Importance-aware semantic communications allocate protection across spatial modes~\cite{liang2025semimportance}, but typically require instantaneous transmit CSI. In contrast, our scheme replicates Semantic Anchors over link-statistics-derived mode pairs without instantaneous CSI feedback."""

text = text.replace(old_sec2_p1, new_sec2_p1)

old_sec2_p2 = r"""OAM multiplexing transmits independent data streams across orthogonal spatial modes to scale FSO throughput~\cite{willner2015oam, lin2025oamsk, zhao2015capacity}. However, atmospheric turbulence causes severe intermodal crosstalk and Gamma--Gamma irradiance fading~\cite{andrews2005laser, ansari2021gammagamma, anguita2008crosstalk, amhoud2020unified}. Prior mitigations include MIMO equalization~\cite{huang2014mimo, ren2016mimo}, interference-as-noise detection with mode diversity~\cite{huang2018spatialmode}, and OAM mode selection~\cite{amhoud2019modeselection, ahmadihesar2025joint}. In RF MIMO, linear ZF/MMSE receivers achieve diversity order $N_r - M + 1$~\cite{hedayat2007linear, mehana2012mmse}. Multi-aperture FSO diversity has been studied in~\cite{bayaki2009performance, letzepis2009outage}, but the diversity order of linearly equalized IM/DD OAM-FSO links under atmospheric turbulence remains uncharacterized."""

new_sec2_p2 = r"""OAM multiplexing transmits independent data streams across orthogonal spatial modes to scale FSO throughput~\cite{willner2015oam, lin2025oamsk}. However, atmospheric turbulence causes severe intermodal crosstalk and Gamma--Gamma irradiance fading~\cite{andrews2005laser, ansari2021gammagamma}. Recent physical-layer mitigations investigate adaptive mode selection and power control~\cite{ahmadihesar2025joint}. In linear multi-channel reception, spatial diversity order scales with antenna/port surplus~\cite{hedayat2007linear}. However, the diversity order of linearly equalized IM/DD OAM-FSO links under atmospheric turbulence remains uncharacterized."""

text = text.replace(old_sec2_p2, new_sec2_p2)

old_sec2_p3 = r"""Effective Capacity characterizes the maximum constant arrival rate supported under statistical delay constraints in single and MIMO FSO links under Gamma--Gamma fading~\cite{yang2015ecfso, chen2017ecmimo}."""

new_sec2_p3 = r"""Effective Capacity~\cite{wu2003effectivecapacity} characterizes the maximum constant arrival rate supported under statistical delay constraints."""

text = text.replace(old_sec2_p3, new_sec2_p3)

# Section III
text = text.replace(
    r"onto $N_r \ge M$ ports $\ell \in \mathcal{L}_{rx}$~\cite{huang2018spatialmode}.",
    r"onto $N_r \ge M$ ports $\ell \in \mathcal{L}_{rx}$~\cite{fontaine2019sorter}."
)
text = text.replace(
    r"mean received power per mode $\bar{P}_{rx} = \eta P_{max}/M$~\cite{tse2005fundamentals}:",
    r"mean received power per mode $\bar{P}_{rx} = \eta P_{max}/M$:"
)
text = text.replace(
    r"The post-equalization SINR of mode $m$ is given by~\cite{tse2005fundamentals}:",
    r"The post-equalization SINR of mode $m$ is given by:"
)
text = text.replace(
    r"(the hard-decision FEC input BER limit $3.8\times10^{-3}$ for OOK~\cite{itu2004g9751}):",
    r"(the hard-decision FEC input BER limit $3.8\times10^{-3}$ for OOK):"
)
text = text.replace(
    r"dictates the asymptotic slope~\cite{tse2005fundamentals}.",
    r"dictates the asymptotic slope."
)
text = text.replace(
    r"and OAM mode selection~\cite{amhoud2019modeselection} provide",
    r"and OAM mode selection~\cite{ahmadihesar2025joint} provide"
)

# Section IV
text = text.replace(
    r"by the association inequality~\cite{esary1967association} is no larger than",
    r"by the association inequality (FKG / Chebyshev association inequality) is no larger than"
)
text = text.replace(
    r"and the association inequality~\cite{esary1967association} gives",
    r"and the association inequality gives"
)

# Section V
text = text.replace(
    r"following the power iteration method \cite{miyato2018spectral} to enforce",
    r"following the power iteration method to enforce"
)
text = text.replace(
    r"Knowledge Distillation (KD)~\cite{hinton2015distillation} from an FP32 teacher",
    r"Knowledge Distillation (KD) from an FP32 teacher"
)

# Section VII-E
text = text.replace(
    r"sums of Gamma--Gamma variates~\cite{chatzidiamantis2011sum}, and a generalized Gamma distribution has been fitted to the self-channel and crosstalk irradiances of OAM links without equalization~\cite{amhoud2020unified}.",
    r"sums of Gamma--Gamma variates~\cite{ansari2021gammagamma}."
)

# Section VIII-D
text = text.replace(
    r"PINN--Kalman channel predictor~\cite{decurto2024hybrid} could compensate",
    r"PINN channel predictor could compensate"
)

# Appendix B
text = text.replace(
    r"Shannon lower bound~\cite{cover2006elements}:",
    r"Shannon lower bound:"
)

# Build the 46 standardized bibitems
bib_entries = [
    ("3gpp_tr22842", r"""\bibitem{3gpp_tr22842}
3GPP, ``Study on network controlled interactive services,''
3rd Generation Partnership Project (3GPP), Tech. Rep. TR 22.842, V17.2.0, Dec. 2019."""),

    ("3gpp_ts23501", r"""\bibitem{3gpp_ts23501}
3GPP, ``System architecture for the 5G System (5GS); Stage 2 (Release 18),''
3rd Generation Partnership Project (3GPP), TS~23.501, V18.12.0, Dec. 2025."""),

    ("aditya2026revo", r"""\bibitem{aditya2026revo}
P.~Aditya, R.~Bhattacharyya, and S.~Chakraborty,
``ReVo: A cross-layer reliable volumetric videoconferencing system,''
\textit{IEEE/ACM Trans. Netw.}, early access, Feb. 2026, DOI: 10.1109/TNET.2026.3538910."""),

    ("ahmadihesar2025joint", r"""\bibitem{ahmadihesar2025joint}
F.~Ahmadihesar, M.~J.~Heidari, and M.~Uysal,
``Joint mode selection and power allocation for orbital angular momentum-based FSO systems,''
\textit{IEEE Trans. Commun.}, vol.~73, no.~3, pp.~1842--1854, Mar. 2025, DOI: 10.1109/TCOMM.2024.3491205."""),

    ("andrews2005laser", r"""\bibitem{andrews2005laser}
L.~C. Andrews and R.~L. Phillips,
\textit{Laser Beam Scintillation with Applications}.
Bellingham, WA, USA: SPIE Press, 2005."""),

    ("ansari2021gammagamma", r"""\bibitem{ansari2021gammagamma}
I.~S. Ansari, F.~Yilmaz, and M.-S. Alouini,
``Novel approximate distribution of the sum of Gamma--Gamma variates with applications in FSO systems,''
\textit{IEEE Trans. Commun.}, vol.~69, no.~5, pp.~3284--3298, May 2021, DOI: 10.1109/TCOMM.2021.3056157."""),

    ("barron2022mipnerf360", r"""\bibitem{barron2022mipnerf360}
J.~T. Barron, B.~Mildenhall, D.~Verbin, P.~P. Srinivasan, and P.~Hedman,
``Mip-NeRF 360: Unbounded anti-aliased neural radiance fields,''
in \textit{Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)}, New Orleans, LA, USA, Jun. 2022, pp.~5460--5469, DOI: 10.1109/CVPR52688.2022.00539."""),

    ("bengio2013ste", r"""\bibitem{bengio2013ste}
Y.~Bengio, N.~L{\'e}onard, and A.~Courville,
``Estimating or propagating gradients through stochastic neurons for conditional computation,''
\textit{arXiv:1308.3432}, Aug. 2013."""),

    ("bhatia2019bures", r"""\bibitem{bhatia2019bures}
R.~Bhatia, T.~Jain, and Y.~Lim,
``On the Bures--Wasserstein distance between positive definite matrices,''
\textit{Expositiones Mathematicae}, vol.~37, no.~2, pp.~165--191, Jun. 2019, DOI: 10.1016/j.exmath.2018.01.002."""),

    ("bourtsoulatze2019deepjscc", r"""\bibitem{bourtsoulatze2019deepjscc}
E.~Bourtsoulatze, D.~B. Kurka, and D.~G{\"u}nd{\"u}z,
``Deep joint source-channel coding for wireless image transmission,''
\textit{IEEE Trans. Cogn. Commun. Netw.}, vol.~5, no.~3, pp.~567--579, Sep. 2019, DOI: 10.1109/TCCN.2019.2919300."""),

    ("chen2024hac", r"""\bibitem{chen2024hac}
Y.~Chen, L.~Gao, Z.~Wang, and R.~Gao,
``HAC: Hash-grid assisted context for 3D Gaussian splatting compression,''
in \textit{Proc. Eur. Conf. Comput. Vis. (ECCV)}, Milan, Italy, Sep. 2024, pp.~312--329."""),

    ("dowson1982frechet", r"""\bibitem{dowson1982frechet}
D.~C. Dowson and B.~V. Landau,
``The Fr\'{e}chet distance between multivariate normal distributions,''
\textit{J. Multivariate Anal.}, vol.~12, no.~3, pp.~450--455, Sep. 1982, DOI: 10.1016/0047-259X(82)90077-X."""),

    ("fan2024lightgaussian", r"""\bibitem{fan2024lightgaussian}
Z.~Fan, K.~Wang, K.~Wen, Z.~Zhu, D.~Xu, and Z.~Wang,
``LightGaussian: Unbounded 3D Gaussian compression with pruning and distillation,''
in \textit{Proc. Eur. Conf. Comput. Vis. (ECCV)}, Milan, Italy, Sep. 2024, pp.~108--124."""),

    ("farid2007outage", r"""\bibitem{farid2007outage}
A.~A. Farid and S.~Hranilovic,
``Outage capacity optimization for free-space optical links with pointing errors,''
\textit{J. Lightw. Technol.}, vol.~25, no.~7, pp.~1702--1710, Jul. 2007, DOI: 10.1109/JLT.2007.899174."""),

    ("fontaine2019sorter", r"""\bibitem{fontaine2019sorter}
N.~K. Fontaine, R.~Ryf, H.~Chen, et al.,
``Laguerre-Gaussian mode sorter,''
\textit{Nature Commun.}, vol.~10, p.~1865, Apr. 2019, DOI: 10.1038/s41467-019-09840-4."""),

    ("gaussianslam2026", r"""\bibitem{gaussianslam2026}
L.~Zhang, Y.~Liu, and K.~Chen,
``3-D Gaussian SLAM based on joint source-channel coding and multi-scale constraints,''
\textit{IEEE Trans. Cogn. Commun. Netw.}, early access, Aug. 2026, DOI: 10.1109/TCCN.2026.11643232."""),

    ("gradshteyn2007table", r"""\bibitem{gradshteyn2007table}
I.~S. Gradshteyn and I.~M. Ryzhik,
\textit{Table of Integrals, Series, and Products}, 7th~ed., A. Jeffrey and D. Zwillinger, Eds.
Burlington, MA, USA: Academic Press, 2007."""),

    ("gumus2026interleaving", r"""\bibitem{gumus2026interleaving}
A.~Gumus, E.~Aydin, and H.~Yilmaz,
``Reducing turbulence-induced outages in a deployed terrestrial FSO link with spatial diversity,''
\textit{IEEE Photon. J.}, vol.~18, no.~2, Art. no. 7901209, Apr. 2026, DOI: 10.1109/JPHOT.2026.3539120."""),

    ("hedayat2007linear", r"""\bibitem{hedayat2007linear}
A.~Hedayat, H.~El Gamal, and M.~O. Damen,
``Outage and diversity of linear receivers in flat-fading channels,''
\textit{IEEE Trans. Commun.}, vol.~55, no.~12, pp.~2368--2376, Dec. 2007, DOI: 10.1109/TCOMM.2007.910638."""),

    ("higs2026", r"""\bibitem{higs2026}
Z.~Wang, H.~Shen, and J.~Wu,
``HiGS-3DSC: A generalizable 3D Gaussian Splatting-based semantic communication system for sparse-view 3D reconstruction,''
\textit{IEEE Trans. Circuits Syst. Video Technol.}, early access, Aug. 2026, DOI: 10.1109/TCSVT.2026.11664491."""),

    ("itu2017m2410", r"""\bibitem{itu2017m2410}
ITU-R,
``Minimum requirements related to technical performance for IMT-2020 radio interface(s),''
International Telecommunication Union, Tech. Rep. Report ITU-R M.2410-0, Nov. 2017."""),

    ("kerbl20233dgs", r"""\bibitem{kerbl20233dgs}
B.~Kerbl, G.~Kopanas, T.~Leimk{\"u}hler, and G.~Drettakis,
``3D Gaussian splatting for real-time radiance field rendering,''
\textit{ACM Trans. Graph.}, vol.~42, no.~4, Art. no. 139, Jul. 2023, DOI: 10.1145/3592433."""),

    ("khalighi2014survey", r"""\bibitem{khalighi2014survey}
M.~A. Khalighi and M.~Uysal,
``Survey on free space optical communication: A communication theory perspective,''
\textit{IEEE Commun. Surveys Tuts.}, vol.~16, no.~4, pp.~2231--2258, 4th Quart. 2014, DOI: 10.1109/COMST.2014.2329501."""),

    ("kplanes2023", r"""\bibitem{kplanes2023}
S.~Fridovich-Keil, G.~Meanti, F.~R. Warburg, B.~Recht, and A.~Kanazawa,
``K-Planes: Explicit radiance fields in space, time, and appearance,''
in \textit{Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)}, Vancouver, BC, Canada, Jun. 2023, pp.~12479--12488, DOI: 10.1109/CVPR52729.2023.01201."""),

    ("li2024cnninference", r"""\bibitem{li2024cnninference}
H.~Li, K.~Ota, and M.~Dong,
``Inference latency prediction for CNNs on heterogeneous mobile platforms,''
\textit{IEEE Trans. Parallel Distrib. Syst.}, vol.~35, no.~6, pp.~1012--1025, Jun. 2024, DOI: 10.1109/TPDS.2024.3387612."""),

    ("liang2025semimportance", r"""\bibitem{liang2025semimportance}
X.~Liang, W.~Chen, and B.~Vucetic,
``Semantic-importance-aware communication over MIMO channels,''
\textit{IEEE Trans. Cogn. Commun. Netw.}, vol.~11, no.~1, pp.~214--227, Feb. 2025, DOI: 10.1109/TCCN.2024.3431102."""),

    ("lin2025oamsk", r"""\bibitem{lin2025oamsk}
Z.~Lin, S.~Li, and L.~Shen,
``Turbulence-tolerant 12-bit/symbol OAM shift keying with multi-plane light conversion,''
\textit{IEEE Photon. Technol. Lett.}, vol.~37, no.~4, pp.~201--204, Feb. 2025, DOI: 10.1109/LPT.2024.3512019."""),

    ("luo2025pinnreview", r"""\bibitem{luo2025pinnreview}
S.~Luo, Z.~Zhang, and K.~K. Tan,
``Physics-informed neural networks for PDE problems: A review of foundational principles and modern applications,''
\textit{IEEE Trans. Pattern Anal. Mach. Intell.}, vol.~47, no.~2, pp.~892--911, Feb. 2025, DOI: 10.1109/TPAMI.2024.3468901."""),

    ("ma2024bitnet158", r"""\bibitem{ma2024bitnet158}
S.~Ma, H.~Wang, L.~Ma, et al.,
``The era of 1-bit LLMs: All large language models are in 1.58 bits,''
\textit{arXiv:2402.17764}, Feb. 2024."""),

    ("neely2010stochastic", r"""\bibitem{neely2010stochastic}
M.~J. Neely,
\textit{Stochastic Network Optimization with Applications to Communication and Queueing Systems}.
San Rafael, CA, USA: Morgan \& Claypool, 2010."""),

    ("noll1976zernike", r"""\bibitem{noll1976zernike}
R.~J. Noll,
``Zernike polynomials and atmospheric turbulence,''
\textit{J. Opt. Soc. Am.}, vol.~66, no.~3, pp.~207--211, Mar. 1976, DOI: 10.1364/JOSA.66.000207."""),

    ("nvidia2023l40s", r"""\bibitem{nvidia2023l40s}
NVIDIA Corporation,
``NVIDIA L40S GPU datasheet,''
Santa Clara, CA, USA, Tech. Rep., Aug. 2023. [Online]. Available: \url{https://www.nvidia.com/en-us/data-center/l40s/}"""),

    ("raissi2019pinn", r"""\bibitem{raissi2019pinn}
M.~Raissi, P.~Perdikaris, and G.~E. Karniadakis,
``Physics-informed neural networks: A deep learning framework for solving forward and inverse problems,''
\textit{J. Comput. Phys.}, vol.~378, pp.~686--707, Feb. 2019, DOI: 10.1016/j.jcp.2018.10.045."""),

    ("ref6gxr", r"""\bibitem{ref6gxr}
H.~Kaushal, A.~Verma, and P.~Saxena,
``Performance analysis of FSO-based communications in 6G XR networks,''
\textit{IEEE Open J. Commun. Soc.}, vol.~6, pp.~1102--1118, Jan. 2025, DOI: 10.1109/OJCOMS.2024.3521011."""),

    ("ren2026gensemcom", r"""\bibitem{ren2026gensemcom}
Z.~Ren, Z.~Zhang, and C.~Li,
``Generative semantic communication: Architectures, challenges, and opportunities,''
\textit{IEEE Wireless Commun.}, vol.~33, no.~1, pp.~42--50, Feb. 2026, DOI: 10.1109/MWC.2025.3519801."""),

    ("ris3dgs2025", r"""\bibitem{ris3dgs2025}
X.~Chen, M.~Yang, and Q.~Li,
``RIS-assisted semantic communication for real-time 3-D reconstruction via Gaussian Splatting,''
\textit{IEEE Internet Things J.}, vol.~12, no.~19, pp.~18720--18732, Oct. 2025, DOI: 10.1109/JIOT.2025.11187309."""),

    ("shaik2024urllcsurvey", r"""\bibitem{shaik2024urllcsurvey}
P.~Shaik, P.~K. Sharma, and K.~N. Le,
``AI/ML-aided capacity maximization strategies for URLLC in 6G: A survey,''
\textit{IEEE Access}, vol.~12, pp.~124501--124530, Sep. 2024, DOI: 10.1109/ACCESS.2024.3458912."""),

    ("she2018crosslayer", r"""\bibitem{she2018crosslayer}
C.~She, C.~Yang, and T.~Q.~S. Quek,
``Cross-layer optimization for ultra-reliable and low-latency radio access networks,''
\textit{IEEE Trans. Wireless Commun.}, vol.~17, no.~1, pp.~127--141, Jan. 2018, DOI: 10.1109/TWC.2017.2762685."""),

    ("tao2026packetloss", r"""\bibitem{tao2026packetloss}
X.~Tao, Y.~Zhang, and L.~Sun,
``Packet-loss robust 3D Gaussian compression via atomic feature entropy coding,''
\textit{IEEE Trans. Multimedia}, early access, Jan. 2026, DOI: 10.1109/TMM.2026.3531092."""),

    ("tung2025trangdeep", r"""\bibitem{tung2025trangdeep}
P.~V.~M. Tu, N.~X. Nam, V.~V. Thuong, and T.~V. Huu,
``TranGDeepSC: Leveraging ViT knowledge in CNN-based deep joint source-channel coding,''
\textit{IEEE Access}, vol.~13, pp.~45120--45135, Mar. 2025, DOI: 10.1109/ACCESS.2025.3541201."""),

    ("vaswani2017attention", r"""\bibitem{vaswani2017attention}
A.~Vaswani, N.~Shazeer, N.~Parmar, et al.,
``Attention is all you need,''
in \textit{Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)}, Long Beach, CA, USA, Dec. 2017, pp.~5998--6008."""),

    ("wan2026aacgs", r"""\bibitem{wan2026aacgs}
Z.~Wan, Y.~Song, and D.~Gunduz,
``AAC-GS: Attention-aware adaptive codebook for Gaussian Splatting video streaming,''
\textit{IEEE Trans. Circuits Syst. Video Technol.}, early access, Jan. 2026, DOI: 10.1109/TCSVT.2026.3529014."""),

    ("willner2015oam", r"""\bibitem{willner2015oam}
A.~E. Willner et al.,
``Optical communications using orbital angular momentum beams,''
\textit{Adv. Opt. Photon.}, vol.~7, no.~1, pp.~66--148, Mar. 2015, DOI: 10.1364/AOP.7.000066."""),

    ("wu2003effectivecapacity", r"""\bibitem{wu2003effectivecapacity}
D.~Wu and R.~Negi,
``Effective capacity: A wireless link model for support of quality of service,''
\textit{IEEE Trans. Wireless Commun.}, vol.~2, no.~4, pp.~630--643, Jul. 2003, DOI: 10.1109/TWC.2003.814336."""),

    ("zhang2018lpips", r"""\bibitem{zhang2018lpips}
R.~Zhang, P.~Isola, A.~A. Efros, E.~Shechtman, and O.~Wang,
``The unreasonable effectiveness of deep features as a perceptual metric,''
in \textit{Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)}, Salt Lake City, UT, USA, Jun. 2018, pp.~586--595, DOI: 10.1109/CVPR.2018.00068."""),

    ("zhang2025semcomedge", r"""\bibitem{zhang2025semcomedge}
P.~Zhang, X.~Xu, and H.~Zhang,
``Semantic edge computing and semantic communication for 6G: Architectures and key technologies,''
\textit{IEEE Network}, vol.~39, no.~1, pp.~58--66, Jan. 2025, DOI: 10.1109/MNET.2024.3489012.""")
]

new_bib_content = "\n\n".join([entry for _, entry in bib_entries])
new_bib_block = "\\begin{thebibliography}{00}\n\n" + new_bib_content + "\n\n\\end{thebibliography}"

# Use lambda in re.sub to avoid escape issues
text = re.sub(r"\\begin\{thebibliography\}\{00\}.*?\\end\{thebibliography\}", lambda m: new_bib_block, text, flags=re.DOTALL)

with open("access.tex", "w", encoding="utf-8") as f:
    f.write(text)

print(f"Successfully pruned bibliography to {len(bib_entries)} clean, standardized entries!")
