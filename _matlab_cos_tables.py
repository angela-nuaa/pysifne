"""Versioned Windows-runtime cosine tables for MATLAB-compatible OFT factors.

The tables were generated on Windows from the runtime used by the verified
MATLAB fixture comparison. They are intentionally limited to Nangles values
covered by the frozen fixtures and the current SIFNE configuration. Linux
libm is not substituted when a table is requested.
"""

from __future__ import annotations

import base64
import functools
import zlib

import numpy as np


TABLE_SOURCE = (
    "Windows Python math.cos / Windows CRT-compatible runtime; "
    "validated against the frozen MATLAB LFT_OFT fixtures"
)

_TABLES_B85: dict[int, str] = {
    4: "c-muN0D%wolQ+j0d)T+{d-C@`5WL_2sx?)uEPtULOq>CSIv9;h9n2hD>Imrv0BIOdCj",
    5: "c-muN0D%wo|L?C~`R>?d`yY$AKbJp#xnDdxO9DhEa~b4*1d8{4{|^N3;p!NO)CUtMRv(Ner0*M49%c_AeasIae3(Db^})=8se|cz@)xXrf8)1&?hnT<@88X_yA`O;elL`+g^I(}0RTs|#U}",
    7: "c-muN0D%wo+jlYST=4m^{U`Ur#!1y@_9rd)`fksaH~Sm^PVxoOe|NsQ)&ms(B7Oe$LZCXBdIlW&Q=t0m(e%UQ(e-bG@_*sb4|5Mp|27=@nLJSS!`u(!|H7evE1LZ<_tvB7hnc?>hyP*WgG>J}9R7du_dgK4-@o)tx))IY{u@mCYx=9t*xv|nnFrK|prPV0bpVZi3D^",
    8: "c-muN0D%woWjE)!&Rg)ze)8r!#vb<V`!**koCNasKl%F~2;T3v^OHR^7s!9r`c$nff1%yiXA6Di1J%RyF_38vOdW1>V!-knaF_!#pImcbG{T%1qRfG*hq(i0&IY2*fw&hI4&<5xiw9ih5DFhm`|cQl)K#FVE5#A70KPyzM*",
    9: "c-muN0D%wo$y4n*@|$1S_uenCkhD5xfB#f`HUq<h{V`p4s=f)n+plc0^!HaF{p4Ij(7*rp_unm9@lg(_PDe4cya1>Vrk{aA^I+;JH4i3Fsd+H|U@;HoKT6Gmg*T<<!Q!7%^Pc?u4+QV`Ke==AKTv%C+SNRwGFHd-7n&@X^8f#Rdz*(2)<ARYt)O%kRD2Iq-E*iu0H)Sw(*",
    10: "c-muN0D%woyE%5ZmOp-J|L6YtmG6#Swr~BG&;8-p<^8`GaeoGi?|<_5KM=g%FOr=l0pedf_TK$1P`-%EAon9s{dvongbJX3m^lnoG8d+gkhzUebAJ#p7iK>$bN4{`|Dp1=1k8o`1Kr#`WSR@3Vdld8^`A&{8_~iS=1!QoF#BqWG#5SIVBtuvx#;l*i)TXSHWG+87$24n37Ly4-q6j(<v->J5cPXWOz!~YuOs*",
    12: "c-muN0D%woYR7NfsM_}0zR+aBl>h(l+fUwn$H>FJec$a<@!1Rv5BBeRcGa|c+w1*L{{9Do_xoQizFiGezyH&@hM<4{@9%%rnyOZoztAqDGM|T$;eq{frpuFn>h1f!gVn>$XCTtPJ!IMkGmqBx!SoYvUyLzG{e2ScgSn4b`*IQfBgsBk{L<P!7=I6jeUR|UCE7lS`r9PhcOT(DqU<A-A7J4}Yx`j7fq45M;X`VEfTeF@?ISh4z~o{1mDcvb${$kgBQ-w|iuWw2K2<b*>LixC00VAaJO",
    13: "c-muN0D%wo+=06<Ef0NTfADqR`>DE*>^trMI9A`@XTNo~b;r$(*Y>MFd!Tpr^Yi^jR!HsnBKUrP`%|%G5Iv3A%Hj-A{FLje3)g_^CYilE-3HWGduz4gbf7r}71Qch0L|_D{vQb5!|h=}v-beh-b$#wJD~PzqS@O9wRZ}U_QL#wZZ8*<zYi*(3AJ|*0edGBuoo6?F#mAT*j`xtz`|`Gk@k`s&aiX<)0au4z57YD7nZJI;xKi4NVJ#ObOlS#FdCP=Fmv|fu$LQ$e{sbx%%5#I?A?K8?*S6i^A5E93ya4oB-*>5#B>Eqm#}mSvv(qi_7WSvuzUy0hoss|Y`Q`(Ct>9bE_<g?DSlymSUF0_URb(9PtUM)0E-`3xZx@%pZo>ew}0+44NahZ`-4*Ec1+WKv|rx#D_i63efzDMigj*myk=ii*1H*KpS{t^8T~+e;B+Ze+zG1gzYRp6Kh&JrP;+7S008U%y#N",
    16: "c-muN0D%wob`rkFvM0Q?FS|L<b>4z!_AGoCRUg&fu%Ep7j<JV*`@VmnrL~W1Z|vWksBjV}zyI{U3f`OvZ}&g>`yUA2??08RSP#^{-^Ne&&|IMWg+n^$9st$9YJI9!mcP($&n(HSPk`oqeYVhNKG6K4#3dm>{cw932FM*Sdj`rKFmngW9WZk!b;mCv-C>934w(BXcSj6S?l_I+4wyPh-N8YoJ7D7Ua>pq&cfj34Wq068ryJ1R0ZSLS-0=(U4v2p^sFY4&?ttYB`ndx&USRovTz62KK49el`R*VseUziQ16B@_>kd-WsXdxIVD$hlcfiVt(`fF1rQZz%(g(~QdmP~bs|V@j4)pNAWzIK5xEg`X;enb{fx|zxP<{zio)MAGOhEb>!?2f!0Q?N13;",
    20: "c-muN0D%wo5)!SveEIL}cXRA+Er0yd{*YwQb*1x<?El<fzw+I&%l4Bu-!b;EZ{OGYEuZ_tvCI2k&pMu~eE!k?-;20E1J&;jEC1^T)W849-~T}He*d*CJXd&ue4*?t36Q+(v^x$!^Y&jm_TK$1Q2(per)p*S3+;-y401mL&7UP{%c26b@4RJ9LIu$N371M6K<<ILi(!zu6XuS=>Q0!wgVmiddj_jJVdhfXosA^AQv`=QVg9AQJ7Y+4=XD(Jgz2M}I}bzE|EH2WVd|*m&OK21BP6;r1cy6e?jLA({=nf*7)`1>Vez<!NOvA4kl$b6a3?JP5OQZD+?^2r!@}bTk?tfmzoWYoR{jjUJF&+*EFV+LorlRRmvE&|SUE^7camE!;mSv_dW3rJBrm`3#o<m^Jxnckwo=)huyza~cf!i^>uBzTl>@N+_JYWKw3o<uhqVI-t2<%sXj-|G*m4P1yc6<&Eh0S`fy`xo05SI{)ZBegb0yKt6^HV7LFErX)&D@`PZN-Nodntq0K*tmYX",
}

SUPPORTED_NANGLES = tuple(sorted(_TABLES_B85))
_ANGLE_COUNTS = {
    4: 4,
    5: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    12: 13,
    13: 14,
    16: 16,
    20: 20,
}


@functools.lru_cache(maxsize=None)
def matlab_windows_factor_table(n_angles: int) -> np.ndarray:
    """Return the frozen MATLAB-compatible factor table for ``Nangles``.

    Rows follow the CUDA orientation-index convention: angle orientations
    first, followed by the zero-orientation row used for unprocessed pixels.
    Columns follow the C-MEX angle traversal order.
    """

    value = int(n_angles)
    if value != n_angles or value not in _TABLES_B85:
        raise ValueError(
            f"No frozen Windows/MATLAB factor table for Nangles={n_angles!r}; "
            f"supported values are {SUPPORTED_NANGLES}"
        )
    payload = zlib.decompress(base64.b85decode(_TABLES_B85[value]))
    angle_count = _ANGLE_COUNTS[value]
    expected_size = (angle_count + 1) * angle_count * 8
    if len(payload) != expected_size:
        raise RuntimeError(
            f"corrupt MATLAB factor table for Nangles={value}: "
            f"{len(payload)} != {expected_size} bytes"
        )
    return np.frombuffer(payload, dtype="<f8").reshape((angle_count + 1, angle_count)).copy()


__all__ = ["SUPPORTED_NANGLES", "TABLE_SOURCE", "matlab_windows_factor_table"]
