// ===== Firebase 로그인 게이트 =====
// 아래 firebaseConfig를 네 Firebase 콘솔(console.firebase.google.com) 프로젝트의
// 설정값으로 반드시 바꿔야 동작한다. 프로젝트 만들고 '웹 앱 추가'하면 이 값이 나온다.
const firebaseConfig = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  projectId: "YOUR_PROJECT",
  storageBucket: "YOUR_PROJECT.appspot.com",
  messagingSenderId: "YOUR_SENDER_ID",
  appId: "YOUR_APP_ID",
};

const FIREBASE_NOT_CONFIGURED = firebaseConfig.apiKey === "YOUR_API_KEY";

function showLogin() {
  document.getElementById("loginScreen").style.display = "flex";
  document.getElementById("appScreen").style.display = "none";
}
function showApp(user) {
  document.getElementById("loginScreen").style.display = "none";
  document.getElementById("appScreen").style.display = "block";
  document.getElementById("userInfo").textContent = user.displayName || user.email || "로그인됨";
  startAppAfterLogin();
}

if (FIREBASE_NOT_CONFIGURED) {
  // Firebase 설정을 아직 안 채웠으면 로그인 없이 바로 앱을 연다 (로컬 테스트/개발 편의용).
  document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("loginScreen").style.display = "none";
    document.getElementById("appScreen").style.display = "block";
    document.getElementById("userInfo").textContent = "(로그인 기능 미설정)";
    startAppAfterLogin();
  });
} else {
  firebase.initializeApp(firebaseConfig);

  document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("googleLoginBtn").addEventListener("click", () => {
      const provider = new firebase.auth.GoogleAuthProvider();
      firebase.auth()
        .signInWithPopup(provider)
        .catch((err) => alert("로그인 실패: " + err.message));
    });
    document.getElementById("logoutBtn").addEventListener("click", () => {
      firebase.auth().signOut();
    });
  });

  firebase.auth().onAuthStateChanged((user) => {
    if (user) showApp(user);
    else showLogin();
  });
}
