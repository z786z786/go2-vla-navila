# P4-T1 解包摘要

- JPEG：601,125
- video 目录：10,819
- JPEG 字节合计：19,236,004,882
- `train/` 磁盘占用（du -sb）：19,280,708,626 字节；文件系统剩余：1,880,441,290,752 字节
- 解包模式：`skipped_existing_complete`；耗时 0.000 秒；脚本总耗时 72.151 秒
- tar↔磁盘双向差集：文件/目录均为空
- 抽样规则：Sort all MNT-relative .jpg paths bytewise in POSIX lexical order; choose floor(i*(N-1)/63), i=0..63.
- 隔离文件：`/mnt/wxh/go2_short_vln/data/navila_dataset/R2R/annotations.INCOMPLETE-184991744B.json.quarantine`，184,991,744 字节（already_quarantined）
- 帧连续性异常：0
- 写权限探针：success，写入 18 字节后删除

## 64 个确定性抽样哈希

| ordinal | sorted_index | path | bytes | sha256 |
|---:|---:|---|---:|---|
| 0 | 0 | `data/navila_dataset/R2R/train/1/frame_0.jpg` | 23700 | `0f05ff03c4e2491d379daba99833bb721d753376f1b76ea16711f6d7aa562a36` |
| 1 | 9541 | `data/navila_dataset/R2R/train/10151/frame_26.jpg` | 21691 | `ccaac12eb1ef84f8053a055fe95669390309965ff2e0967f90e0cdba9cabdc29` |
| 2 | 19083 | `data/navila_dataset/R2R/train/10304/frame_1.jpg` | 44508 | `44f97552b6a0dd72ddb749174e41494b36c15ff356bb774a0311e3d657900172` |
| 3 | 28624 | `data/navila_dataset/R2R/train/10469/frame_12.jpg` | 23585 | `909bf35d6226c29523e42e16ba793f439b45e239591b3e86d308dc45daa127a8` |
| 4 | 38166 | `data/navila_dataset/R2R/train/10620/frame_27.jpg` | 35579 | `d5dbca99fcb0a8664db524c0a9c3d734cd772a7b7d7eed3259f84ab6a1c0ce3a` |
| 5 | 47708 | `data/navila_dataset/R2R/train/10779/frame_13.jpg` | 37554 | `9a587dc3fda388de2ed399d91fca5d987e15bab68ce420ffe6842873d44e6e40` |
| 6 | 57249 | `data/navila_dataset/R2R/train/1183/frame_27.jpg` | 28566 | `4b5bcd5d463c05ca7b7207ff991083f56304425185909540b5db5e8da6b3fb85` |
| 7 | 66791 | `data/navila_dataset/R2R/train/1331/frame_4.jpg` | 18999 | `98a1d7791c15719189f74bfba4be60ecc1fbfd711a77486ed2095936c3a66355` |
| 8 | 76333 | `data/navila_dataset/R2R/train/1488/frame_36.jpg` | 22257 | `7c6c63a378efb30ef134ca56e4d3e7c13e8491b5b6240e7fef29bb789c29ad18` |
| 9 | 85874 | `data/navila_dataset/R2R/train/1636/frame_59.jpg` | 28969 | `3ea113b3cb9cbe70c1629bafded5daa08bc722b95f6a669e56ed06b59be12062` |
| 10 | 95416 | `data/navila_dataset/R2R/train/18/frame_36.jpg` | 45001 | `b254f2c3bb65947bff42ac6c31133c75d2913fa7b9c8e31a1d7a81e801280d7f` |
| 11 | 104958 | `data/navila_dataset/R2R/train/1957/frame_10.jpg` | 19326 | `6ffa959f454a67f632969b43fd06e66c552088c3b43380c6f7f6f567d5140f60` |
| 12 | 114499 | `data/navila_dataset/R2R/train/2120/frame_34.jpg` | 34898 | `996aad86308f39c5076c59ed0cab7e6f9cd499ed38e677732b6619459f837d8a` |
| 13 | 124041 | `data/navila_dataset/R2R/train/2288/frame_30.jpg` | 29395 | `ae6b8443622310bf85b59496127b8dce4ea71d2ba0f15f3a9053a5ff15d2ad37` |
| 14 | 133583 | `data/navila_dataset/R2R/train/2430/frame_14.jpg` | 42697 | `0e139cda1572140859b2275703cfda168c58d63f44c1cd55678cca6c477bbf4d` |
| 15 | 143124 | `data/navila_dataset/R2R/train/2589/frame_45.jpg` | 19309 | `21a3d9d763e744b540660e1ea5a17d999c1bb838139a44cbe66f79294a3be73d` |
| 16 | 152666 | `data/navila_dataset/R2R/train/2740/frame_40.jpg` | 27616 | `8bfb43180caad8e8de3dafb415ae21e2b1d2480b64ed591708b6f4f8f16388eb` |
| 17 | 162208 | `data/navila_dataset/R2R/train/2896/frame_26.jpg` | 22395 | `d7abef6a1646d6d29a90aa2434e9b74d7be97edd2e7d82958f71a7de592f2c8d` |
| 18 | 171749 | `data/navila_dataset/R2R/train/3045/frame_29.jpg` | 17045 | `5b78b0c35410e3af73fca4d7e38e3e2fa772ab8c03d4abca42169cf4f8f482f0` |
| 19 | 181291 | `data/navila_dataset/R2R/train/3189/frame_144.jpg` | 23168 | `fca212a4de70af08d8ccd116e39c9136412a7f52416cb2d4035687921739eb8a` |
| 20 | 190833 | `data/navila_dataset/R2R/train/3348/frame_9.jpg` | 34049 | `f1220ead15c33a35ad9b1aa454bb3294e8adb8c23f8f2f3f4cb1951049623285` |
| 21 | 200374 | `data/navila_dataset/R2R/train/3500/frame_44.jpg` | 29983 | `bae79e21a9e49955b8e093cad67b51b1f91e1f1c422f4373274c689e63c2589a` |
| 22 | 209916 | `data/navila_dataset/R2R/train/3664/frame_33.jpg` | 32470 | `dc9983342ade087f00044f6a0323e672039c0f89492cffca7a57f9eeff3567d0` |
| 23 | 219457 | `data/navila_dataset/R2R/train/3819/frame_42.jpg` | 31110 | `a70b248d8e617df8914ee525f47e8454f1b8e93c55f0b76b35a050643e54df2b` |
| 24 | 228999 | `data/navila_dataset/R2R/train/3974/frame_38.jpg` | 20651 | `fcd1d9f7515cd58b4ba249cc977d544819fbb4935b481d7ce3ef6e245b65da96` |
| 25 | 238541 | `data/navila_dataset/R2R/train/4134/frame_14.jpg` | 49588 | `a74ea5217512cb18560b65807fa69d37a76f522f3b70f07e1731370ad343c21b` |
| 26 | 248082 | `data/navila_dataset/R2R/train/4280/frame_24.jpg` | 17588 | `86072517ab4b74f1527f66d01542715ad1fbbc40320125c973c09c3f10464a9e` |
| 27 | 257624 | `data/navila_dataset/R2R/train/443/frame_40.jpg` | 52662 | `ac30180d60f8adaa8f6d7cbc03754aca38ffafa30a57839fce08c121698393c1` |
| 28 | 267166 | `data/navila_dataset/R2R/train/4586/frame_22.jpg` | 22532 | `2ace35ad06612f2fa6aaf9d12e3dde925934bd1709a9272e6e140c9bbfa5a282` |
| 29 | 276707 | `data/navila_dataset/R2R/train/4732/frame_6.jpg` | 21659 | `a41f56fc7f3d5837f2d8dd67851019f03a242ec33c6b4bcddcb86e180f04e3d2` |
| 30 | 286249 | `data/navila_dataset/R2R/train/4858/frame_27.jpg` | 27394 | `4681f5864311411fc35ac12b5e436e511f6f5b0cc8a9da8bd5bc9238b7065d81` |
| 31 | 295791 | `data/navila_dataset/R2R/train/502/frame_26.jpg` | 17327 | `59c8bd39ee190e10f49e0b4e48ed758e41395eeedd6e4349c1e143bcfbae03b6` |
| 32 | 305332 | `data/navila_dataset/R2R/train/5175/frame_21.jpg` | 32470 | `2c406e2ac0bd11662e2ddbcd41d3de54e7928f13dba2f2ea97c5763d77568a10` |
| 33 | 314874 | `data/navila_dataset/R2R/train/5333/frame_15.jpg` | 31044 | `342304490f3ad2e4b44ca4debbc159d0c1576095b8a9a9f525828236fb982891` |
| 34 | 324416 | `data/navila_dataset/R2R/train/5486/frame_14.jpg` | 34578 | `24b2f495d90fe396afbce08a06e4b06b2bdc37660d9dbe477af34637ca9a6c10` |
| 35 | 333957 | `data/navila_dataset/R2R/train/5647/frame_26.jpg` | 32158 | `8ec7d8320a4c8bfb4c81b20f9b9bd5a0e3dddc2924feb45f302a8e6e656f5753` |
| 36 | 343499 | `data/navila_dataset/R2R/train/5801/frame_14.jpg` | 28406 | `f4a8591ebb256d9078a3d03d844a5eb64058686beb95e7d1fc259c961cc575d3` |
| 37 | 353041 | `data/navila_dataset/R2R/train/5968/frame_15.jpg` | 40320 | `f3aad293ba91db82111f09200151f470f2a4e755fb7357e43c3098ababb56088` |
| 38 | 362582 | `data/navila_dataset/R2R/train/6113/frame_32.jpg` | 21720 | `a17a659e124bcd76b451a6bc1132b73d649f19a8f42dee23428d80ae8a7427c7` |
| 39 | 372124 | `data/navila_dataset/R2R/train/6263/frame_7.jpg` | 24864 | `a1f8a59bb6138dc3ab56c75270cda9c619f99973f17e60f58a98432510f6ff6f` |
| 40 | 381666 | `data/navila_dataset/R2R/train/643/frame_19.jpg` | 39778 | `707ae89ea84f44d14d4662e70efff940344f6d2742c8edfb310c5ddf548ec19b` |
| 41 | 391207 | `data/navila_dataset/R2R/train/6585/frame_27.jpg` | 27258 | `64bcc3fcd5e91aed335ed9bfd3b19bfd4e9c43cca14abdfc2a53f3c60b211f27` |
| 42 | 400749 | `data/navila_dataset/R2R/train/6723/frame_8.jpg` | 20859 | `660a59b1f04155f49b2f4cd1f173e8975c7fbe27b5cdb0e27b1e3a5e0e4c9d1d` |
| 43 | 410290 | `data/navila_dataset/R2R/train/688/frame_9.jpg` | 23155 | `d5a60f7ed592c845b218351d28c606b845a81d22c729f8b6868a0daa782d2bb4` |
| 44 | 419832 | `data/navila_dataset/R2R/train/7028/frame_32.jpg` | 32490 | `97483d239b0419cc84a5d7a3a2eee225ac9d56ad6aabbee95b794a56b2e39321` |
| 45 | 429374 | `data/navila_dataset/R2R/train/7186/frame_69.jpg` | 35746 | `8493f6a28f944ff68556f3f8eca5b74ddcd0aaa54d02f64c68a71acf9761e195` |
| 46 | 438915 | `data/navila_dataset/R2R/train/7351/frame_19.jpg` | 36115 | `684a25241c9713517d94c2534532c10cd92fae10e052cc51db4a3118f0f116c1` |
| 47 | 448457 | `data/navila_dataset/R2R/train/7501/frame_13.jpg` | 27006 | `b3ee5877b242fb2cfb026770292e1b47294683ed6ca13de538443f0912b38c04` |
| 48 | 457999 | `data/navila_dataset/R2R/train/7652/frame_2.jpg` | 41344 | `3be07fbbb0bbb52f782d48b006f1647b9bd791ea55c2a9b91531c0bbd6ffad91` |
| 49 | 467540 | `data/navila_dataset/R2R/train/781/frame_38.jpg` | 67184 | `b6f2a2966a8bdae5ffe9816fbdb01936d8cf15a2ab4d50cc42c8ca4dab03ebce` |
| 50 | 477082 | `data/navila_dataset/R2R/train/7975/frame_1.jpg` | 38229 | `ccadf50eabd0a8751d31566c8d7cb9ebc157dd1e019c93228d6ab6e7279f44d3` |
| 51 | 486624 | `data/navila_dataset/R2R/train/814/frame_38.jpg` | 26743 | `4b2174a1593eb064b633042b7f67b4170459f136660bed3a2877fd9d7c96454c` |
| 52 | 496165 | `data/navila_dataset/R2R/train/8291/frame_18.jpg` | 39545 | `cfd17cd958c639ef653240100fe8dbfd9bda4ae7a966896a1f2874efda1bdd00` |
| 53 | 505707 | `data/navila_dataset/R2R/train/8440/frame_6.jpg` | 25077 | `250d858c55cd377b0bf9d26f137102f6afc26fa0489b192f2d1b6ec127f5d8b0` |
| 54 | 515249 | `data/navila_dataset/R2R/train/8602/frame_18.jpg` | 39870 | `89fc142bf6478a19346f9689beae78cc8a03a87cd6480dcbfd56d8a5fcd51c60` |
| 55 | 524790 | `data/navila_dataset/R2R/train/875/frame_33.jpg` | 46466 | `0a4aedbb38c794b6902917bc47a7014ff2aa6462a6b2083bdbecde695f3442c3` |
| 56 | 534332 | `data/navila_dataset/R2R/train/8899/frame_11.jpg` | 22596 | `97136ef4637c5492c72eeb77e67cd5d219f3af9f8e4490be5939bf0a71d66241` |
| 57 | 543874 | `data/navila_dataset/R2R/train/9042/frame_74.jpg` | 29427 | `7686f297e6183677420920a3b1cdebbd2d0049dd967e0884f0d41767b81fa1ee` |
| 58 | 553415 | `data/navila_dataset/R2R/train/9201/frame_50.jpg` | 29351 | `f019725fbeb3c0d1c3b0cb59824d3f32b4b024e23f32613abdb4183b2e9dacde` |
| 59 | 562957 | `data/navila_dataset/R2R/train/9370/frame_26.jpg` | 38782 | `10a358701ec38c928a41850fdc0c02b15b1368594e3b3e1622dacd20be50c86e` |
| 60 | 572499 | `data/navila_dataset/R2R/train/9529/frame_71.jpg` | 24650 | `945b86d259a59ca8a4764416af3519869310a174cdd276a689c296da6b905963` |
| 61 | 582040 | `data/navila_dataset/R2R/train/9689/frame_39.jpg` | 33218 | `f77c951b0ebb03f2a4e2f01820dbb250a99ae74ceedda5e4b5924f3ac975b8d0` |
| 62 | 591582 | `data/navila_dataset/R2R/train/9849/frame_33.jpg` | 26275 | `9481c27550f08297a529b84f2942b4a348013debe92c988355cbfd5513ccf79f` |
| 63 | 601124 | `data/navila_dataset/R2R/train/9999/frame_9.jpg` | 39785 | `604e9a55534f7f493d8fd04f46e1bf2a958c250f345d89da4a4c63af3be721c1` |

## 未完成项

- 无（MISSING：无）
