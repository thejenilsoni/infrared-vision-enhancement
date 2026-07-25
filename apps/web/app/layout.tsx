import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "IRIS | Infrared Interpretation Suite",
  description:
    "Infrared enhancement, perceptual colorization, and uncertainty-aware object interpretation."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
