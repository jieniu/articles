/**
 * Welcome to your Workbox-powered service worker!
 *
 * You'll need to register this file in your web app and you should
 * disable HTTP caching for this file too.
 * See https://goo.gl/nhQhGp
 *
 * The rest of the code is auto-generated. Please don't update this file
 * directly; instead, make changes to your Workbox build configuration
 * and re-run your build process.
 * See https://goo.gl/2aRDsh
 */

importScripts("https://storage.googleapis.com/workbox-cdn/releases/3.6.3/workbox-sw.js");

/**
 * The workboxSW.precacheAndRoute() method efficiently caches and responds to
 * requests for URLs in the manifest.
 * See https://goo.gl/S9QRab
 */
self.__precacheManifest = [
  {
    "url": "404.html",
    "revision": "e9a28e46ad0f576a73bcec1d24a05b3b"
  },
  {
    "url": "AI/index.html",
    "revision": "7837b286a2398f4f6c67e4fb7b2040b2"
  },
  {
    "url": "AI/ready_for_machine_learning.html",
    "revision": "3f0cabec09d12bcabd6ad5b4882e52e7"
  },
  {
    "url": "AI/reverse-integer.html",
    "revision": "cd8e13f4513781d086df1f03f4496531"
  },
  {
    "url": "assets/css/0.styles.14fa570c.css",
    "revision": "b93b7781945fe9d7de2098e88a088390"
  },
  {
    "url": "assets/img/search.83621669.svg",
    "revision": "83621669651b9a3d4bf64d1a670ad856"
  },
  {
    "url": "assets/js/10.870aafa0.js",
    "revision": "05631991daeb6ad368c0ddf01526c635"
  },
  {
    "url": "assets/js/11.ab2a631a.js",
    "revision": "acfe4a5afbdf321f92e04ab886755819"
  },
  {
    "url": "assets/js/12.eb0521f2.js",
    "revision": "7ea17e2a379c394e27f0f62939266645"
  },
  {
    "url": "assets/js/13.6cc4e31e.js",
    "revision": "d17ba674592a3c8b25d79734942216d2"
  },
  {
    "url": "assets/js/14.f1391346.js",
    "revision": "a1502bf2fe7fbb32b5d86e2102028126"
  },
  {
    "url": "assets/js/15.693a2a52.js",
    "revision": "adcbf283c9e1c034b4baffab6bc01c37"
  },
  {
    "url": "assets/js/16.6a6236a3.js",
    "revision": "3232465b0910b8d832c66f28b252eb3a"
  },
  {
    "url": "assets/js/17.072abdd2.js",
    "revision": "83014116b9b3a84a6109ef06fa0b2e5c"
  },
  {
    "url": "assets/js/18.459c8d32.js",
    "revision": "517479f56e1ea47c8b47a8ed26bf7af9"
  },
  {
    "url": "assets/js/19.b0fa80c2.js",
    "revision": "9a248a514decc7249a784b5d5a1ea94a"
  },
  {
    "url": "assets/js/2.9f734292.js",
    "revision": "0b0cd1f7b5e95ff2dbe95d1caef95945"
  },
  {
    "url": "assets/js/20.e1cc392a.js",
    "revision": "043b0ef6fb5696b00c771c7bf47b3667"
  },
  {
    "url": "assets/js/21.bc4ced03.js",
    "revision": "0590ca5bc3e639856fd67545b8cf0479"
  },
  {
    "url": "assets/js/22.15e90473.js",
    "revision": "7d4690134465290827cba1d8302dbaae"
  },
  {
    "url": "assets/js/23.1e842e22.js",
    "revision": "49953a424af1a76dc43ec253b9a15ff9"
  },
  {
    "url": "assets/js/24.89be4fb3.js",
    "revision": "d4f9e2ba58186671777f69daecf608f4"
  },
  {
    "url": "assets/js/25.c0a4f010.js",
    "revision": "0877542891277cbe1dbafcfc0c2a58aa"
  },
  {
    "url": "assets/js/26.c1daf4e9.js",
    "revision": "000688e1576192779c70558401749935"
  },
  {
    "url": "assets/js/3.02ea72ff.js",
    "revision": "c4cb75990c5aebb274fcbaaa8be5b785"
  },
  {
    "url": "assets/js/4.d7229487.js",
    "revision": "e0c6c0b154e8203918082914d521db78"
  },
  {
    "url": "assets/js/5.4837771a.js",
    "revision": "b84dbf69edd0f2565c539c2e15b13a79"
  },
  {
    "url": "assets/js/6.5f57bf07.js",
    "revision": "ca1c11e96cb3286abcefe9a322d414b5"
  },
  {
    "url": "assets/js/7.f50df1bf.js",
    "revision": "7ae5c8345fdbc5a912ffb2c20fadb20e"
  },
  {
    "url": "assets/js/8.3a66950f.js",
    "revision": "5176e165376990a9cef1b705a6ed39e0"
  },
  {
    "url": "assets/js/9.795a67d8.js",
    "revision": "edfd6309ba20f5149c7fe7db6aa5ee24"
  },
  {
    "url": "assets/js/app.12b95bc1.js",
    "revision": "006c1ca4b360548cd121eb8485bfe5a9"
  },
  {
    "url": "cpp/assignment_to_self_in_assignment_operator.html",
    "revision": "0861d2b68e2a19f3b12f4565409f4edf"
  },
  {
    "url": "cpp/exception_and_destructor.html",
    "revision": "dbd72efa4e6dd23a6bd23fcd0da00847"
  },
  {
    "url": "cpp/index.html",
    "revision": "07a3ce1d1913beefffcf6bffa9903dab"
  },
  {
    "url": "cpp/RAII.html",
    "revision": "d58277b974f85aa457276f2fd18670df"
  },
  {
    "url": "cpp/static_initialization_fiasco.html",
    "revision": "6ab495fe6c4587ad35dc8f75fcf281c2"
  },
  {
    "url": "cpp/struct-and-class.html",
    "revision": "9d6a966823a219166f85674dbcdd5156"
  },
  {
    "url": "cpp/virtual_destructor.html",
    "revision": "b9379d30e9690d5e9b35c427eb071bfa"
  },
  {
    "url": "cpp/virtual_function_and_constructor.html",
    "revision": "e41e4953030ea8dffd7d69cd829ea49a"
  },
  {
    "url": "home.jpg",
    "revision": "281f1b0f5331799f77adc6e480159e88"
  },
  {
    "url": "home.png",
    "revision": "5895b4975a13b932f5484c2a4d20bfe1"
  },
  {
    "url": "image-20190105220419469.png",
    "revision": "725c9d5341a07708f655bcfc541dd157"
  },
  {
    "url": "image-20190105223100319.png",
    "revision": "3e69e4c78d5513f8e79a6c28dbae44af"
  },
  {
    "url": "index.html",
    "revision": "7bb248aa36775cd21ceb94edf1d033f5"
  },
  {
    "url": "java/index.html",
    "revision": "e5109e31ca8a83bca5ccb117adbe356d"
  },
  {
    "url": "java/powermock_and_unittest.html",
    "revision": "4c6d49458210ed825ca1b94ded4cc180"
  },
  {
    "url": "java/spring_boot_thread_pool_timer.html",
    "revision": "67992b6d8f4944266c358904451998f7"
  },
  {
    "url": "java/understanding_collections_threadsafe.html",
    "revision": "d44501aa6b69ae6ed0a7473ff8c8e740"
  },
  {
    "url": "jupyter_new.png",
    "revision": "dfd535d58a25bf87f39daecdf923571b"
  },
  {
    "url": "jupyter_start.png",
    "revision": "89f56ae3b76fdb480652126bac0f96d0"
  },
  {
    "url": "math/index.html",
    "revision": "0095cf98e581921a43fc2ce21bfe4585"
  },
  {
    "url": "math/linear_transformation.html",
    "revision": "450c23c2ec4daf76ccccd99f6f1e26b5"
  },
  {
    "url": "mysql_notes/1.MySQL架构.html",
    "revision": "566d89a37b082b0ac3076e7d4b9c0897"
  },
  {
    "url": "mysql_notes/index.html",
    "revision": "ef6ab06878e673a8bfb3edc561f6b12a"
  },
  {
    "url": "mysql_notes/MySQL 配置汇总.html",
    "revision": "02a2e4d05c1b0fee54f762fdff83bb5c"
  },
  {
    "url": "tools/index.html",
    "revision": "09d89f6bc7e25cca3aa573da11aa2000"
  },
  {
    "url": "tools/vuepress_website.html",
    "revision": "fd2cd20417d904177f9c4ccfe69ac81f"
  }
].concat(self.__precacheManifest || []);
workbox.precaching.suppressWarnings();
workbox.precaching.precacheAndRoute(self.__precacheManifest, {});
addEventListener('message', event => {
  const replyPort = event.ports[0]
  const message = event.data
  if (replyPort && message && message.type === 'skip-waiting') {
    event.waitUntil(
      self.skipWaiting().then(
        () => replyPort.postMessage({ error: null }),
        error => replyPort.postMessage({ error })
      )
    )
  }
})
