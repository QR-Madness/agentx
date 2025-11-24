import axios, {AxiosRequestConfig, AxiosResponse} from "axios";
import {api} from "../lib/api";

const urls = {
    getQuickChatConversations: () => `${api.baseUrl}/quick-chats`,
    getQuickChatConversation: (conversationId: string) => `${api.baseUrl}/quick-chats/${conversationId}`
}

export interface QuickChatMessage {
    sender: string;
    content: string;
}

export interface QuickChatConversation {
    id: string;
    messages: QuickChatMessage[];
}

const getQuickChatConversations = async (config: AxiosRequestConfig): Promise<QuickChatConversation[]> => {
    const res = await axios.get(urls.getQuickChatConversations(), config);
    return res.data;
}