// import { useEffect, useState } from "react";

// const KEY = "cognify-demo-session";

// export type DemoSession = { name: string; email: string };

// function read(): DemoSession | null {
//   if (typeof window === "undefined") return null;
//   try {
//     const raw = window.localStorage.getItem(KEY);
//     return raw ? (JSON.parse(raw) as DemoSession) : null;
//   } catch {
//     return null;
//   }
// }

// export function signIn(session: DemoSession) {
//   window.localStorage.setItem(KEY, JSON.stringify(session));
//   window.dispatchEvent(new Event("cognify-auth"));
// }

// export function signOut() {
//   window.localStorage.removeItem(KEY);
//   window.dispatchEvent(new Event("cognify-auth"));
// }

// /** Returns undefined while hydrating, then the session or null. */
// export function useDemoSession(): DemoSession | null | undefined {
//   const [session, setSession] = useState<DemoSession | null | undefined>(undefined);

//   useEffect(() => {
//     const sync = () => setSession(read());
//     sync();
//     window.addEventListener("cognify-auth", sync);
//     window.addEventListener("storage", sync);
//     return () => {
//       window.removeEventListener("cognify-auth", sync);
//       window.removeEventListener("storage", sync);
//     };
//   }, []);

//   return session;
// }
